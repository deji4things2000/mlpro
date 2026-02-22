"""
Deep inspection of AdroitHandRelocate-v1
Our primary environment for differential dynamics
"""
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import mujoco

gym.register_envs(gymnasium_robotics)

ENV_ID = "AdroitHandRelocate-v1"
print(f"\n{'=' * 65}")
print(f"  Deep Inspection: {ENV_ID}")
print(f"{'=' * 65}")

env = gym.make(ENV_ID, render_mode=None)
obs, info = env.reset(seed=42)

# ── 1. Spaces ────────────────────────────────────────────────────────────────
print(f"\n📐 ACTION SPACE")
print(f"   Shape  : {env.action_space.shape}  ({env.action_space.shape[0]} DOFs)")
print(f"   Low    : {env.action_space.low[:6].round(3)} ...")
print(f"   High   : {env.action_space.high[:6].round(3)} ...")
print(f"   Dtype  : {env.action_space.dtype}")

print(f"\n👁️  OBSERVATION SPACE")
print(f"   Shape  : {env.observation_space.shape}")
print(f"   Low    : {env.observation_space.low[:6].round(3)} ...")
print(f"   High   : {env.observation_space.high[:6].round(3)} ...")

print(f"\n📊 INITIAL OBSERVATION (first reset)")
print(f"   obs[:10] = {obs[:10].round(4)}")

# ── 2. MuJoCo Model Internals ────────────────────────────────────────────────
print(f"\n🔧 MUJOCO MODEL INTERNALS")
model = env.unwrapped.model
data  = env.unwrapped.data

print(f"   n_bodies    (nbody)   : {model.nbody}")
print(f"   n_joints    (njnt)    : {model.njnt}")
print(f"   n_actuators (nu)      : {model.nu}")
print(f"   n_geoms     (ngeom)   : {model.ngeom}")
print(f"   n_sensors   (nsensor) : {model.nsensor}")
print(f"   qpos shape            : {data.qpos.shape}  (joint positions)")
print(f"   qvel shape            : {data.qvel.shape}  (joint velocities)")
print(f"   ctrl shape            : {data.ctrl.shape}  (actuator controls)")

# Joint names
print(f"\n🦾 JOINT NAMES")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = ["free","ball","slide","hinge"][model.jnt_type[i]]
    print(f"   [{i:>2}] {name:<30} type={jtype}")

# Actuator names  
print(f"\n⚙️  ACTUATOR NAMES")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"   [{i:>2}] {name}")

# ── 3. Jacobian Structure ────────────────────────────────────────────────────
print(f"\n📐 JACOBIAN STRUCTURE (key for differential dynamics)")
nv = model.nv   # degrees of freedom in velocity space
print(f"   nv (velocity DOFs)    : {nv}")
print(f"   Jacobian shape (3xnv) : (3, {nv})")
print(f"   Full Jac shape        : (6, {nv})  [3 trans + 3 rot]")

# Compute jacobian for a body (e.g. fingertip or palm)
# Find end-effector body
ee_candidates = ["palm", "forearm", "fftip", "mftip", "rftip", "lftip", "thtip"]
print(f"\n   End-effector bodies found:")
for name in ee_candidates:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr])
        rank = np.linalg.matrix_rank(J)
        print(f"   ✅  {name:<12} body_id={body_id}  J={J.shape}  rank={rank}")

# ── 4. Reward & Task Structure ───────────────────────────────────────────────
print(f"\n🎯 TASK & REWARD STRUCTURE")
rewards = []
obs, _ = env.reset(seed=0)
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        obs, _ = env.reset()

rewards = np.array(rewards)
print(f"   Reward range  : [{rewards.min():.4f}, {rewards.max():.4f}]")
print(f"   Reward mean   : {rewards.mean():.4f}")
print(f"   Reward std    : {rewards.std():.4f}")
print(f"   Info keys     : {list(info.keys())}")
print(f"   Is sparse?    : {np.all(np.isin(np.unique(rewards.round(2)), [0, 1, -1]))}")

# ── 5. State Space Breakdown ─────────────────────────────────────────────────
print(f"\n🗂️  OBSERVATION BREAKDOWN (39 dims)")
obs, _ = env.reset(seed=0)
print(f"   Full obs: {obs.round(3)}")
print(f"\n   Likely breakdown for Adroit Relocate:")
print(f"   [0:27]  = hand joint angles (27 DOFs)")
print(f"   [27:33] = object position + orientation (6D)")  
print(f"   [33:36] = target position (3D)")
print(f"   [36:39] = object-target delta (3D)")
print(f"   Total   = 39 dims ✅")

env.close()

print(f"\n{'=' * 65}")
print(f"✅  Inspection complete — ready for differential dynamics!")
print(f"{'=' * 65}\n")
