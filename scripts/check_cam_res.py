"""
Verify that ROBOMME_CAMERA_RES is propagating through to SAPIEN — peek at
the actual camera width/height after env build.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train.envs.rl_env import RobommeRLEnv

print(f"ROBOMME_OBS_MODE   = {os.environ.get('ROBOMME_OBS_MODE')}")
print(f"ROBOMME_CAMERA_RES = {os.environ.get('ROBOMME_CAMERA_RES')}")

env = RobommeRLEnv(env_id="BinFill", seed=0, shape_reward=False)
env.reset()

inner = env._env.unwrapped
print(f"\nlive cameras:")
for name, cfg in inner._sensor_configs.items():
    w = getattr(cfg, "width", "?")
    h = getattr(cfg, "height", "?")
    print(f"  {name:20s} {w} x {h}")
env.close()
