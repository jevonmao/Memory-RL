"""Optional Weights & Biases logging for the RL baselines.

Pass --wandb on any train_ppo*.py script to mirror every SB3 TensorBoard
scalar (the train/ and rollout/ metrics) into a W&B run. W&B is an optional
dependency: nothing here imports `wandb` unless --wandb is set.
"""

from __future__ import annotations


def add_wandb_args(parser):
    """Register the shared --wandb* flags on an argparse parser."""
    g = parser.add_argument_group("wandb")
    g.add_argument("--wandb", action="store_true",
                   help="Log this run to Weights & Biases.")
    g.add_argument("--wandb_project", default="robomme-benchmark")
    g.add_argument("--wandb_entity", default=None,
                   help="W&B team/user; defaults to the logged-in account.")
    g.add_argument("--wandb_name", default=None,
                   help="Run name; defaults to <baseline>_<task>.")
    return parser


def init_wandb(args, baseline, config):
    """Init a W&B run and return a WandbCallback, or None if --wandb is unset.

    `sync_tensorboard=True` pipes every SB3 TB scalar into W&B automatically,
    so no per-metric plumbing is needed.
    """
    if not getattr(args, "wandb", False):
        return None

    import wandb
    from wandb.integration.sb3 import WandbCallback

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"{baseline}_{args.task}",
        group=args.task,
        tags=[baseline, args.task],
        config=config,
        sync_tensorboard=True,
        save_code=True,
        dir=args.outdir,
    )
    return WandbCallback(verbose=1)


def finish_wandb(args):
    """Close the active W&B run (no-op if --wandb was not set)."""
    if getattr(args, "wandb", False):
        import wandb
        wandb.finish()
