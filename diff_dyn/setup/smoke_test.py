import gymnasium as gym
import gymnasium_robotics
import numpy as np
import mujoco
import sys

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)
print(f"Python path     : {sys.executable}")
print(f"MuJoCo          : {mujoco.__version__}")
print(f"Gymnasium       : {gym.__version__}")
print(f"Gym-Robotics    : {gymnasium_robotics.__version__}")

gym.register_envs(gymnasium_robotics)
all_envs = list(gym.envs.registry.keys())
print(f"Total envs      : {len(all_envs)}")

allegro = [e for e in all_envs if "llegro" in e]
adroit  = [e for e in all_envs if "Adroit" in e]
shadow  = sorted([e for e in all_envs if "Hand" in e])

print(f"\nAllegro envs    : {allegro or '❌ None'}")
print(f"Adroit envs     : {adroit or '❌ None'}")
print(f"Shadow Hand envs: {shadow[:6]}")

PRIORITY = [
    "HandReach-v3",
    "HandManipulateBlock-v1",
    "HandManipulateBlockRotateZ-v1",
    "AdroitHandDoor-v1",
    "AdroitHandHammer-v1",
    "AdroitHandPen-v1",
    "AdroitHandRelocate-v1",
    "FetchReach-v3",
]

print("\n" + "=" * 60)
print("SMOKE TESTS (v1/v2/v3 only)")
print("=" * 60)

working = []
for env_id in PRIORITY:
    if env_id not in all_envs:
        print(f"  ⏭️  SKIP  {env_id}  (not registered)")
        continue
    try:
        env = gym.make(env_id, render_mode=None)
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            obs, rew, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
        dof = env.action_space.shape[0]
        obs_shape = (
            {k: v.shape for k, v in obs.items()}
            if isinstance(obs, dict) else obs.shape
        )
        env.close()
        print(f"  ✅  {env_id:<45} {dof:>3} DOFs  obs={obs_shape}")
        working.append((env_id, dof))
    except Exception as e:
        print(f"  ❌  {env_id:<45} {str(e)[:70]}")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
if working:
    best = max(working, key=lambda x: x[1])
    print(f"  Working envs : {len(working)}")
    print()
    for env_id, dof in sorted(working, key=lambda x: -x[1]):
        bar = "█" * (dof // 2)
        print(f"    {dof:>3} DOFs  {bar}  {env_id}")
    print(f"\n  🏆 Best pick : {best[0]}  ({best[1]} DOFs)")
    print(f"  ✅ Ready to build!")
else:
    print("  ❌ Nothing worked — see errors above")
print("=" * 60)
