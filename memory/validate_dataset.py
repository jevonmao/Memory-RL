import h5py
import numpy as np
from collections import defaultdict

# -----------------------------
# Helpers
# -----------------------------
def timestep_index(name):
    """Sort timestep_2 before timestep_10."""
    return int(name.split("_")[-1])


def read_h5_value(x):
    """Read an h5py.Dataset handle into a numpy-compatible value."""
    if isinstance(x, h5py.Dataset):
        return x[()]
    return x


def is_valid_action(action):
    """Check if action is usable for training."""
    if action is None:
        return False, "None"

    action = read_h5_value(action)

    if isinstance(action, (str, bytes)):
        return False, "string"

    try:
        action = np.asarray(action)
    except (TypeError, ValueError):
        return False, f"type={type(action)}"

    if action.size == 0:
        return False, "empty"

    if not np.issubdtype(action.dtype, np.number):
        return False, f"non-numeric dtype={action.dtype}"

    if np.any(~np.isfinite(action)):
        return False, "NaN or Inf"

    return True, None


def scan_episode(ep_grp):
    bad = []
    total = 0

    timesteps = sorted(
        [k for k in ep_grp.keys() if k.startswith("timestep_")],
        key=timestep_index,
    )

    for ts_name in timesteps:
        ts = ep_grp[ts_name]

        if "action" not in ts:
            bad.append((ts_name, "missing action"))
            continue

        act = ts["action"]

        if ACTION_KEY not in act:
            bad.append((ts_name, "missing joint_action"))
            continue

        action = act[ACTION_KEY]
        ok, reason = is_valid_action(action)

        total += 1
        if not ok:
            bad.append((ts_name, reason))

    return total, bad


# -----------------------------
# Config
# -----------------------------
OBS_KEYS = ("eef_state", "joint_state", "gripper_state")
ACTION_KEY = "joint_action"


# -----------------------------
# Main validator
# -----------------------------
def validate(h5_path, max_episodes=None):
    print(f"\nScanning: {h5_path}\n")

    stats = defaultdict(int)
    bad_examples = []

    with h5py.File(h5_path, "r") as hf:

        ep_names = list(hf.keys())

        if max_episodes:
            ep_names = ep_names[:max_episodes]

        for i, ep_name in enumerate(ep_names):

            if i % 50 == 0:
                print(f"  progress: {i}/{len(ep_names)} episodes")

            ep_grp = hf[ep_name]

            if not isinstance(ep_grp, h5py.Group):
                continue

            total, bad = scan_episode(ep_grp)

            stats["episodes"] += 1
            stats["timesteps"] += total
            stats["bad_timesteps"] += len(bad)

            if bad:
                stats["bad_episodes"] += 1

                for item in bad[:5]:  # cap logging noise
                    bad_examples.append((ep_name, item))

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n==================== RESULTS ====================")
    print(f"Episodes scanned     : {stats['episodes']}")
    print(f"Total timesteps      : {stats['timesteps']}")
    print(f"Bad timesteps        : {stats['bad_timesteps']}")
    print(f"Bad episodes         : {stats['bad_episodes']}")

    print("\n==================== SAMPLE ERRORS ====================")

    for ep, (ts, reason) in bad_examples[:20]:
        print(f"[{ep}] {ts} -> {reason}")

    print("\n======================================================\n")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python validate_dataset.py /path/to/file.h5 [max_episodes]")

    path = sys.argv[1]
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    validate(path, max_episodes=max_episodes)  # fast mode by default