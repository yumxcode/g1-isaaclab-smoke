#!/usr/bin/env python
"""Route-b smoke: probe IsaacLab 2.2.1 image, then create the official G1 velocity
env and step it. Success criteria: G1 task registered, env created, >=100 steps."""
import argparse
import subprocess
import sys

print("=== PHASE 0: ENV PROBE ===")
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(),
          "device_count:", torch.cuda.device_count())
except Exception as e:
    print("torch import FAILED:", repr(e))
    sys.exit(2)
try:
    r = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        low = line.lower()
        if any(k in low for k in ["isaac", "omni", "rsl", "mujoco", "onnx", "gym"]):
            print("  PKG:", line.strip())
except Exception as e:
    print("pip list failed:", repr(e))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="IsaacLab G1 smoke")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("=== PHASE 1: APP LAUNCHED ===")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# register tasks (2.x: config package __init__ performs gym.register)
ok_import = False
try:
    from isaaclab_tasks.manager_based import locomotion  # noqa: F401,E402
    ok_import = True
    print("imported isaaclab_tasks.manager_based.locomotion")
except Exception as e:
    print("locomotion import failed:", repr(e))
try:
    import isaaclab_tasks.manager_based.locomotion.velocity.config.g1  # noqa: F401,E402
    ok_import = True
    print("imported g1 config package")
except Exception as e:
    print("g1 config import failed:", repr(e))

g1_ids = sorted(i for i in gym.registry if "G1" in i)
print("=== G1 TASKS REGISTERED ===")
for i in g1_ids:
    print("  ", i)
if not g1_ids:
    print("NO G1 TASKS REGISTERED -> route-b smoke FAILED at registration")
    simulation_app.close()
    sys.exit(3)

task = "Isaac-Velocity-Flat-G1-v0"
if task not in g1_ids:
    task = g1_ids[0]
print("=== PHASE 2: CREATE ENV", task, "===")
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(task)
env_cfg.scene.num_envs = 64
env_cfg.sim.device = str(args_cli.device) if hasattr(args_cli, "device") else "cuda:0"
print("num_envs:", env_cfg.scene.num_envs, "device:", env_cfg.sim.device)

env = gym.make(task, cfg=env_cfg)
print("env created:", env)
obs, _ = env.reset()
print("reset ok. obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))

import time  # noqa: E402

t0 = time.time()
n_steps = 100
for i in range(n_steps):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
dt = time.time() - t0
print("=== STEP LOOP OK: {} steps in {:.1f}s ({:.1f} env-steps/s) ===".format(
    n_steps, dt, n_steps * 64 / dt))
print("reward mean:", float(rew.mean()))
env.close()
simulation_app.close()
print("=== ROUTE-B SMOKE PASSED ===")
