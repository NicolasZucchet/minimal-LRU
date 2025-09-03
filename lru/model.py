from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def setup(self):
        self.theta_log = self.param(
            "theta_log", partial(theta_init, max_phase=self.max_phase), (self.d_hidden,)
        )
        self.nu_log = self.param(
            "nu_log", partial(nu_init, r_min=self.r_min, r_max=self.r_max), (self.d_hidden,)
        )
        self.gamma_log = self.param("gamma_log", gamma_log_init, (self.nu_log, self.theta_log))

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))

    def __call__(self, inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
        C = self.C_re + 1j * self.C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(hidden_states, inputs)

        return outputs


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    lru: LRU  # lru module
    d_model: int  # model size
    dropout: float = 0.0  # dropout probability
    norm: str = "layer"  # which normalization to use
    prenorm: bool = False  # use prenorm or postnorm
    training: bool = True  # in training mode (for dropout)

    def setup(self):
        """Initializes the LRU, normalization and dropout"""
        self.seq = self.lru()
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)
        if self.norm in ["layer"]:
            self.normalization = nn.LayerNorm()
        else:
            self.normalization = nn.BatchNorm(
                use_running_average=not self.training, axis_name="batch"
            )
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not self.training)

    def __call__(self, inputs):
        x = inputs
        if self.prenorm:
            x = self.normalization(x)  # pre normalization
        x = self.seq(x)  # call LRU
        x = self.drop(nn.gelu(x))
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        x = self.drop(x)
        x = inputs + x  # skip connection
        if not self.prenorm:
            x = self.normalization(x)
        return x


class StackedEncoderModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    lru: LRU
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    norm: str = "batch"
    prenorm: bool = False

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                lru=self.lru,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                norm=self.norm,
                prenorm=self.prenorm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, inputs):
        x = self.encoder(inputs)  # embed input in latent space
        for layer in self.layers:
            x = layer(x)  # apply each layer
        return x


class ClassificationModel(nn.Module):
    """Stacked encoder with pooling and softmax"""

    lru: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    pooling: str = "mean"  # pooling mode
    norm: str = "batch"  # type of normalization
    prenorm: bool = False  # use prenorm or postnorm
    multidim: int = 1  # number of outputs

    def setup(self):
        self.encoder = StackedEncoderModel(
            lru=self.lru,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout=self.dropout,
            training=self.training,
            norm=self.norm,
            prenorm=self.prenorm,
        )
        self.decoder = nn.Dense(self.d_output * self.multidim)

    def __call__(self, x):
        x = self.encoder(x)
        if self.pooling in ["mean"]:
            x = jnp.mean(x, axis=0)  # mean pooling across time
        elif self.pooling in ["last"]:
            x = x[-1]  # just take last
        elif self.pooling in ["none"]:
            x = x  # do not pool at all
        x = self.decoder(x)
        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)
        return nn.log_softmax(x, axis=-1)


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)
