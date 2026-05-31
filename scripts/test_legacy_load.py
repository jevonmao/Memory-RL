"""Smoke test: load ppo_v4 (legacy dual-ResNet18 encoder) via the auto-detect helper."""
import sys
from train.evaluate_trained import _load_ppo_with_encoder_autodetect

p = sys.argv[1] if len(sys.argv) > 1 else r"runs\ppo_v4\ppo_BinFill_final.zip"
m = _load_ppo_with_encoder_autodetect(p, "cpu")
print(f"loaded {type(m).__name__}; extractor={type(m.policy.features_extractor).__name__}")
print(f"num_timesteps={m.num_timesteps}")
