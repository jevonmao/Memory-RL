"""Verify PPO.load works on a PPOWithICM-saved checkpoint."""
import sys
from stable_baselines3 import PPO

p = sys.argv[1] if len(sys.argv) > 1 else r"runs\ppo_icm_v7\ppo_icm_BinFill_final.zip"
m = PPO.load(p, device="cpu")
print(f"load OK | class={type(m).__name__} | num_timesteps={m.num_timesteps}")
print(f"policy={type(m.policy).__name__}  device={m.device}")
