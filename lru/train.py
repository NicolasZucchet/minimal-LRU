from functools import partial
from jax import random
import wandb
from .train_helpers import (
    create_train_state,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    reduce_lr_on_plateau,
    train_epoch,
    validate,
)
from .dataloading import Datasets
from .model import BatchClassificationModel
from .model import LRU


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
        )
    else:
        wandb.init(mode="offline")

    lr = args.lr_base
    ssm_lr = args.lr_factor * lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]
    # Dataset dependent logic
    if args.dataset == "copy-classification":
        assert args.pooling == "none", "No pooling for copy task"
        dense_targets = True
    else:
        dense_targets = False
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        if args.dataset in ["aan-classification"]:
            # Use retrieval model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False
    else:
        retrieval = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=args.jax_seed, batch_size=args.batch_size)
    print(f"[*] Starting training on `{args.dataset}` =>> Initializing...")

    lru = partial(
        LRU, d_hidden=args.d_hidden, d_model=args.d_model, r_min=args.r_min, r_max=args.r_max
    )
    if retrieval:
        raise NotImplementedError("Retrieval model not implemented yet")
    else:
        model_cls = partial(
            BatchClassificationModel,
            lru=lru,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.p_dropout,
            pooling=args.pooling,
            norm=args.norm,
            prenorm=args.prenorm,
            multidim=1 + dense_targets,
        )

    # Initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        in_dim=in_dim,
        batch_size=args.batch_size,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        norm=args.norm,
        ssm_lr=ssm_lr,
        lr=lr,
    )

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size / args.batch_size)
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        if epoch < args.warmup_end:
            print("Using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end
        elif args.cosine_anneal:
            print("Using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("Using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state, skey, model_cls, trainloader, seq_len, in_dim, args.norm, lr_params
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state, model_cls, valloader, seq_len, in_dim, args.norm)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state, model_cls, testloader, seq_len, in_dim, args.norm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- Test Loss: {test_loss:.5f}\n"
                f"\tVal Accuracy: {val_acc:.4f} "
                f"-- Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state, model_cls, testloader, seq_len, in_dim, args.norm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
                f"\tTest Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
            input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min
        )

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        metrics = {
            "Training Loss": train_loss,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Count": count,
            "Learning rate count": lr_count,
            "Opt acc": opt_acc,
            "lr": state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"],
            "ssm_lr": state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"],
        }
        if valloader is not None:
            metrics["Test Loss"] = test_loss
            metrics["Test Accuracy"] = test_acc
        wandb.log(metrics)

        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            break
