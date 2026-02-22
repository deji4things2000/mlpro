#!/usr/bin/env python3
"""Verification script for robotics research stack."""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any, Dict, Tuple


def _format_exc() -> str:
    return "".join(traceback.format_exception(*sys.exc_info())).strip()


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_import(name: str) -> Tuple[bool, Any, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, module, str(version)
    except Exception:
        return False, None, _format_exc()


def _check_imports(results: Dict[str, str]) -> Dict[str, Any]:
    _print_header("Import and version checks")
    checks: Dict[str, Any] = {}

    for name in [
        "torch",
        "gymnasium",
        "gymnasium_robotics",
        "mujoco",
        "pinocchio",
        "numpy",
        "pybind11",
    ]:
        ok, module, info = _safe_import(name)
        checks[name] = {"ok": ok, "module": module, "info": info}
        if ok:
            print(f"{name}: version {info}")
        else:
            print(f"{name}: FAILED\n{info}")
            results[name] = info

    torch_mod = checks["torch"]["module"]
    if checks["torch"]["ok"]:
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
            print(f"torch: CUDA available = {cuda_available}")
        except Exception:
            err = _format_exc()
            print(f"torch: CUDA check FAILED\n{err}")
            results["torch_cuda"] = err

    return checks


def _check_gym_envs(results: Dict[str, str], checks: Dict[str, Any]) -> None:
    _print_header("Gymnasium-Robotics environment checks")

    gym_ok = checks.get("gymnasium", {}).get("ok", False)
    gym_robotics_ok = checks.get("gymnasium_robotics", {}).get("ok", False)

    if not (gym_ok and gym_robotics_ok):
        results["gym_envs"] = "gymnasium or gymnasium_robotics import failed"
        print("Skipping env checks due to import failures.")
        return

    gymnasium = checks["gymnasium"]["module"]

    env_ids = ["FetchReach-v3", "FetchPush-v3", "HandManipulateBlock-v1"]

    for env_id in env_ids:
        try:
            env = gymnasium.make(env_id)
            obs, _ = env.reset()
            obs_shape = getattr(obs, "shape", None)
            act_shape = getattr(env.action_space, "shape", None)
            print(f"{env_id}: obs shape = {obs_shape}, action shape = {act_shape}")

            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                _ = (reward, terminated, truncated, info)

            env.close()
        except Exception:
            err = _format_exc()
            print(f"{env_id}: FAILED\n{err}")
            results[f"env:{env_id}"] = err


def _load_panda_model(pinocchio_mod: Any) -> Tuple[Any, Any, str]:
    # Try example_robot_data first, then fall back to pinocchio's examples.
    try:
        example_robot_data = importlib.import_module("example_robot_data")
        robot = example_robot_data.load("panda")
        return robot.model, robot.data, "example_robot_data"
    except Exception:
        try:
            model = pinocchio_mod.buildSampleModelManipulator()
            data = model.createData()
            return model, data, "pinocchio.buildSampleModelManipulator"
        except Exception:
            raise


def _check_pinocchio(results: Dict[str, str], checks: Dict[str, Any]) -> None:
    _print_header("Pinocchio model and dynamics checks")

    if not checks.get("pinocchio", {}).get("ok", False):
        results["pinocchio"] = "pinocchio import failed"
        print("Skipping Pinocchio checks due to import failure.")
        return

    pinocchio_mod = checks["pinocchio"]["module"]

    try:
        model, data, source = _load_panda_model(pinocchio_mod)
        print(f"Loaded Panda model via {source}")
    except Exception:
        err = _format_exc()
        print(f"Failed to load Panda model\n{err}")
        results["pinocchio_model"] = err
        return

    try:
        num_joints = model.njoints
        dof = model.nv
        joint_names = list(model.names)
        print(f"Number of joints: {num_joints}")
        print(f"DOF (nv): {dof}")
        print("Joint names:")
        for name in joint_names:
            print(f"  - {name}")
    except Exception:
        err = _format_exc()
        print(f"Failed to read model info\n{err}")
        results["pinocchio_model_info"] = err
        return

    try:
        import numpy as np

        q = pinocchio_mod.randomConfiguration(model)
        v = np.random.randn(model.nv)
        a = np.random.randn(model.nv)

        tau = pinocchio_mod.rnea(model, data, q, v, a)
        M = pinocchio_mod.crba(model, data, q)

        print(f"RNEA output shape: {getattr(tau, 'shape', None)}")
        print(f"CRBA mass matrix shape: {getattr(M, 'shape', None)}")

        # Ensure symmetry for numerical stability before Cholesky.
        M = 0.5 * (M + M.T)

        try:
            np.linalg.cholesky(M)
            print("CRBA mass matrix is positive definite: True")
        except np.linalg.LinAlgError:
            print("CRBA mass matrix is positive definite: False")
            results["pinocchio_crba_pd"] = "CRBA mass matrix not positive definite"
    except Exception:
        err = _format_exc()
        print(f"Pinocchio dynamics computation FAILED\n{err}")
        results["pinocchio_dynamics"] = err


def main() -> int:
    results: Dict[str, str] = {}

    checks = _check_imports(results)
    _check_gym_envs(results, checks)
    _check_pinocchio(results, checks)

    _print_header("Summary")
    if not results:
        print("ALL CHECKS PASSED")
        return 0

    print("Some checks failed:")
    for key, err in results.items():
        print(f"- {key}: {err.splitlines()[-1]}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
