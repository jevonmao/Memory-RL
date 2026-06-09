from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from memory.dataset import RoboVLAPTPDataset
    from memory.model import SimpleVLA
except ModuleNotFoundError:
    from dataset import RoboVLAPTPDataset
    from model import SimpleVLA


ACTION_KEYS = {
    "action",
    "actions",
    "act",
    "acts",
    "target_action",
    "target_actions",
}


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def find_action_datasets(group: h5py.Group, prefix: str = "") -> list[str]:
    """Return dataset paths that look like action datasets by name."""
    paths: list[str] = []

    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            paths.extend(find_action_datasets(item, path))
        elif isinstance(item, h5py.Dataset):
            normalized = key.lower()
            if normalized in ACTION_KEYS or "action" in normalized:
                paths.append(path)

    return paths


def copy_schema_with_replacements(
    src: h5py.Group,
    dst: h5py.Group,
    replacements: dict[str, np.ndarray],
    prefix: str = "",
    max_items: int | None = None,
) -> None:
    copy_attrs(src.attrs, dst.attrs)

    for key, item in src.items():
        path = f"{prefix}/{key}" if prefix else key

        if isinstance(item, h5py.Group):
            child = dst.create_group(key)
            copy_schema_with_replacements(item, child, replacements, path, max_items)
            continue

        if not isinstance(item, h5py.Dataset):
            continue

        data = replacements.get(path)
        if data is None:
            data = item[...]
            if max_items is not None and data.ndim >= 1 and data.shape[0] >= max_items:
                data = data[:max_items]

        dataset_kwargs: dict[str, Any] = {}
        if item.compression is not None:
            dataset_kwargs["compression"] = item.compression
        if item.compression_opts is not None:
            dataset_kwargs["compression_opts"] = item.compression_opts
        if item.shuffle:
            dataset_kwargs["shuffle"] = item.shuffle
        if item.fletcher32:
            dataset_kwargs["fletcher32"] = item.fletcher32

        dset = dst.create_dataset(key, data=data, **dataset_kwargs)
        copy_attrs(item.attrs, dset.attrs)



def get_state_dict_from_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    cleaned_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key.removeprefix("module.")
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


def build_model_for_checkpoint(state_dict: dict[str, torch.Tensor], device: torch.device) -> SimpleVLA:
    """Instantiate SimpleVLA using dimensions inferred from checkpoint weights."""
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(SimpleVLA)

    if "state_enc.weight" in state_dict:
        hidden_dim = int(state_dict["state_enc.weight"].shape[0])
        state_dim = int(state_dict["state_enc.weight"].shape[1])

        if "hidden_dim" in signature.parameters:
            kwargs["hidden_dim"] = hidden_dim
        if "d_model" in signature.parameters:
            kwargs["d_model"] = hidden_dim
        if "state_dim" in signature.parameters:
            kwargs["state_dim"] = state_dim

    if "policy_head.2.weight" in state_dict:
        action_dim = int(state_dict["policy_head.2.weight"].shape[0])
        if "action_dim" in signature.parameters:
            kwargs["action_dim"] = action_dim

    if "memory.memory" in state_dict:
        memory_slots = int(state_dict["memory.memory"].shape[0])
        if "memory_slots" in signature.parameters:
            kwargs["memory_slots"] = memory_slots
        if "num_memory_tokens" in signature.parameters:
            kwargs["num_memory_tokens"] = memory_slots

    print(f"Instantiating SimpleVLA with inferred args: {kwargs}")
    return SimpleVLA(**kwargs).to(device)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> SimpleVLA:
    state_dict = get_state_dict_from_checkpoint(checkpoint_path, device)
    model = build_model_for_checkpoint(state_dict, device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"[warn] Unexpected checkpoint keys: {unexpected}")

    model.eval()
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.unsqueeze(0).to(device)
        else:
            moved[key] = value
    return moved


def filter_batch_for_model(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    """Adapt RoboVLAPTPDataset sample keys to SimpleVLA.forward() keys."""
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    allowed_keys = set(parameters.keys())

    adapted = dict(batch)

    # RoboVLAPTPDataset returns history_* names, while SimpleVLA.forward expects
    # states/images with shapes (B, T, state_dim) and (B, T, 3, H, W).
    if "states" in allowed_keys and "states" not in adapted and "history_states" in adapted:
        adapted["states"] = adapted["history_states"]
    if "images" in allowed_keys and "images" not in adapted and "history_images" in adapted:
        adapted["images"] = adapted["history_images"]

    # Some datasets may use current_* for single-step inference. If history is
    # unavailable, add a time dimension so SimpleVLA still receives B,T,... tensors.
    if "states" in allowed_keys and "states" not in adapted and "current_state" in adapted:
        current_state = adapted["current_state"]
        if torch.is_tensor(current_state) and current_state.ndim == 2:
            current_state = current_state.unsqueeze(1)
        adapted["states"] = current_state
    if "images" in allowed_keys and "images" not in adapted and "current_image" in adapted:
        current_image = adapted["current_image"]
        if torch.is_tensor(current_image) and current_image.ndim == 4:
            current_image = current_image.unsqueeze(1)
        adapted["images"] = current_image

    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )
    if accepts_kwargs:
        filtered = {
            key: value
            for key, value in adapted.items()
            if key not in {"episode_id", "timestep", "index", "file_id"}
        }
    else:
        filtered = {
            key: value
            for key, value in adapted.items()
            if key in allowed_keys
        }

    missing_required = [
        name
        for name, param in parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in filtered
    ]
    if missing_required:
        raise RuntimeError(
            f"Could not build model batch. Missing required forward args: {missing_required}. "
            f"Available dataset keys: {sorted(batch.keys())}. "
            f"Adapted keys: {sorted(adapted.keys())}."
        )

    dropped = sorted(set(batch.keys()) - set(filtered.keys()))
    if dropped and not getattr(filter_batch_for_model, "_printed_dropped", False):
        print(f"[info] Dropping non-forward batch keys: {dropped}")
        filter_batch_for_model._printed_dropped = True

    return filtered


def extract_action_prediction(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("actions", "action", "pred_actions", "pred_action", "action_pred"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value

    raise RuntimeError(
        "Could not find an action tensor in the model output. "
        "Update extract_action_prediction() to match SimpleVLA.forward()."
    )


def scalar_to_int(value: Any) -> int:
    """Convert dataset metadata values such as tensors/arrays/scalars to int."""
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def normalize_predicted_action(pred: torch.Tensor) -> np.ndarray:
    """Return one single-step action vector from a model output tensor."""
    if pred.ndim >= 1 and pred.shape[0] == 1:
        pred = pred.squeeze(0)

    # Some model outputs may still contain a time dimension. The policy action
    # should correspond to the current / last history step.
    if pred.ndim == 2:
        pred = pred[-1]

    if pred.ndim != 1:
        raise RuntimeError(f"Expected one action vector, got prediction shape {tuple(pred.shape)}")

    return pred.detach().cpu().numpy()


# Helper to resolve action dataset path for a prediction
def resolve_action_path(
    src: h5py.File,
    episode_id: int,
    timestep: int,
    action_dataset_name: str,
    pred_shape: tuple[int, ...],
) -> str | None:
    """Find the target per-timestep action dataset for one prediction."""
    preferred_path = f"episode_{episode_id}/timestep_{timestep}/action/{action_dataset_name}"
    if preferred_path in src:
        return preferred_path

    action_group_path = f"episode_{episode_id}/timestep_{timestep}/action"
    if action_group_path not in src:
        return None

    action_group = src[action_group_path]
    if not isinstance(action_group, h5py.Group):
        return None

    for key, item in action_group.items():
        if isinstance(item, h5py.Dataset) and item.shape == pred_shape:
            candidate_path = f"{action_group_path}/{key}"
            print(
                f"[warn] {preferred_path} not found; using shape-compatible dataset "
                f"{candidate_path} instead."
            )
            return candidate_path

    return None


def infer_action_replacements(
    template_path: Path,
    checkpoint_path: Path,
    history_len: int,
    device: torch.device,
    max_samples: int | None = None,
    action_dataset_name: str = "joint_action",
) -> dict[str, np.ndarray]:
    dataset = RoboVLAPTPDataset([str(template_path)], history_len=history_len)
    model = load_checkpoint(checkpoint_path, device)

    replacements: dict[str, np.ndarray] = {}
    skipped_missing_actions = 0
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Generating predictions for {num_samples}/{len(dataset)} dataset samples")

    with h5py.File(template_path, "r") as src, torch.no_grad():
        for idx in range(num_samples):
            sample = dataset[idx]

            if "episode_id" not in sample or "timestep" not in sample:
                raise RuntimeError(
                    "RoboVLAPTPDataset samples must include episode_id and timestep so predictions "
                    "can be written back into the matching per-timestep h5 action dataset."
                )

            episode_id = scalar_to_int(sample["episode_id"])
            timestep = scalar_to_int(sample["timestep"])

            batch = move_batch_to_device(sample, device)
            model_batch = filter_batch_for_model(model, batch)
            output = model(**model_batch)
            pred = extract_action_prediction(output)
            pred_np = normalize_predicted_action(pred)

            action_path = resolve_action_path(
                src=src,
                episode_id=episode_id,
                timestep=timestep,
                action_dataset_name=action_dataset_name,
                pred_shape=pred_np.shape,
            )
            if action_path is None:
                if not getattr(infer_action_replacements, "_printed_missing_action", False):
                    print(
                        "[warn] Some dataset samples do not have a matching per-timestep action "
                        "dataset in the template h5. These samples will be skipped."
                    )
                    infer_action_replacements._printed_missing_action = True
                skipped_missing_actions += 1
                continue

            template_shape = src[action_path].shape
            if pred_np.shape != template_shape:
                raise RuntimeError(
                    f"Prediction for {action_path} has shape {pred_np.shape}, but template dataset "
                    f"has shape {template_shape}."
                )

            replacements[action_path] = pred_np.astype(src[action_path].dtype, copy=False)

            if (idx + 1) % 100 == 0 or idx + 1 == num_samples:
                print(f"Generated predictions for {idx + 1}/{num_samples} samples")

    print(f"Prepared replacements for {len(replacements)} per-timestep action datasets")
    if skipped_missing_actions:
        print(f"Skipped {skipped_missing_actions} samples without matching action datasets")
    if not replacements:
        raise RuntimeError(
            "No action replacements were prepared. Check whether episode_id/timestep metadata "
            "from RoboVLAPTPDataset matches the h5 template structure."
        )
    return replacements


def make_action_replacements(
    template_path: Path,
    predicted_actions: np.ndarray,
    allow_prefix_match: bool = False,
) -> dict[str, np.ndarray]:
    replacements: dict[str, np.ndarray] = {}

    with h5py.File(template_path, "r") as src:
        action_paths = find_action_datasets(src)

        if not action_paths:
            raise RuntimeError(
                "No action-like dataset was found in the template h5. "
                "Expected a dataset name containing 'action'."
            )

        print(f"Found action-like datasets: {action_paths}")

        for path in action_paths:
            template_shape = src[path].shape

            if predicted_actions.shape == template_shape:
                replacements[path] = predicted_actions.astype(src[path].dtype, copy=False)
                print(f"Replacing {path}: {template_shape}")
            elif (
                allow_prefix_match
                and len(template_shape) == len(predicted_actions.shape)
                and template_shape[0] >= predicted_actions.shape[0]
                and template_shape[1:] == predicted_actions.shape[1:]
            ):
                replacements[path] = predicted_actions.astype(src[path].dtype, copy=False)
                print(
                    f"Replacing prefix of {path}: template shape {template_shape}, "
                    f"replacement shape {predicted_actions.shape}"
                )
            else:
                print(
                    f"[warn] Not replacing {path}: template shape {template_shape} "
                    f"does not match predicted action shape {predicted_actions.shape}"
                )

    if not replacements:
        raise RuntimeError(
            "Generated actions did not match any action dataset shape in the template. "
            "This usually means RoboVLAPTPDataset returns window-level samples while the h5 "
            "stores episode-level actions. Inspect the template action shape and adapt the "
            "aggregation logic accordingly."
        )

    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an h5 file matching the RoboMME demo schema by copying a template "
            "demo h5 and replacing action datasets with actions predicted by a checkpoint."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/data/robomme_data_h5/record_dataset_BinFill.h5"),
        help="Existing RoboMME demo h5 whose schema should be copied.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained SimpleVLA checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data/robomme_data_h5/generated_checkpoint_BinFill.h5"),
        help="Output h5 path.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=8,
        help="History length used by RoboVLAPTPDataset during training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only run inference on the first N RoboVLAPTPDataset samples. Useful for debugging.",
    )
    parser.add_argument(
        "--slice-output",
        action="store_true",
        help=(
            "Write only the first N rows for datasets whose first dimension is at least N. "
            "Use this with --max-samples when generating a smaller debug h5."
        ),
    )
    parser.add_argument(
        "--action-dataset-name",
        type=str,
        default="joint_action",
        help=(
            "Per-timestep action dataset to replace under each action group. "
            "For SimpleVLA checkpoints this should usually be joint_action because ACTION_DIM=8."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference, e.g. cuda or cpu.",
    )
    args = parser.parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"Template h5 not found: {args.template}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading template: {args.template}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Using device: {device}")

    replacements = infer_action_replacements(
        template_path=args.template,
        checkpoint_path=args.checkpoint,
        history_len=args.history_len,
        device=device,
        max_samples=args.max_samples,
        action_dataset_name=args.action_dataset_name,
    )

    output_max_items = None
    with h5py.File(args.template, "r") as src, h5py.File(args.output, "w") as dst:
        copy_schema_with_replacements(
            src,
            dst,
            replacements,
            max_items=output_max_items,
        )

    print(f"Wrote checkpoint-generated h5 file to: {args.output}")


if __name__ == "__main__":
    main()