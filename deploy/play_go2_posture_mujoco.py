import os
import time
import sys


def _ensure_libgomp_preload():
    if os.environ.get("GO2_LIBGOMP_PRELOAD_READY") == "1":
        return
    try:
        machine = os.uname().machine.lower()
    except AttributeError:
        machine = ""
    if "aarch64" not in machine and "arm64" not in machine:
        os.environ["GO2_LIBGOMP_PRELOAD_READY"] = "1"
        return

    preload = os.environ.get("LD_PRELOAD", "")
    gomp_candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        gomp_candidates.append(os.path.join(conda_prefix, "lib", "libgomp.so.1"))
    gomp_candidates.append("/usr/lib/aarch64-linux-gnu/libgomp.so.1")
    gldispatch_candidates = [
        "/lib/aarch64-linux-gnu/libGLdispatch.so.0",
        "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0",
    ]

    libs_to_preload = []
    for candidate in gomp_candidates:
        if candidate and os.path.exists(candidate):
            libs_to_preload.append(candidate)
            break
    for candidate in gldispatch_candidates:
        if candidate and os.path.exists(candidate):
            libs_to_preload.append(candidate)
            break

    if all(candidate in preload.split() for candidate in libs_to_preload):
        os.environ["GO2_LIBGOMP_PRELOAD_READY"] = "1"
        return

    if libs_to_preload:
        env = os.environ.copy()
        env["GO2_LIBGOMP_PRELOAD_READY"] = "1"
        env["LD_PRELOAD"] = " ".join(libs_to_preload + ([preload] if preload else []))
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

    os.environ["GO2_LIBGOMP_PRELOAD_READY"] = "1"


_ensure_libgomp_preload()

from go2_posture_policy_wrapper import Go2PosturePolicyWrapper

import mujoco
import numpy as np

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr


KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_SPACE = 32
KEY_LEFT_CTRL = 341
KEY_RIGHT_CTRL = 345
KEY_0 = 48
KEY_A = 65
KEY_D = 68
KEY_E = 69
KEY_Q = 81
KEY_R = 82
KEY_S = 83
KEY_W = 87


def install_keyboard_command_callback(env):
    lin_step = float(os.environ.get("MUJOCO_KEY_LIN_STEP", "0.1"))
    yaw_step = float(os.environ.get("MUJOCO_KEY_YAW_STEP", "0.2"))
    yaw_sign = float(os.environ.get("MUJOCO_YAW_SIGN", "-1.0"))
    max_lin = float(os.environ.get("MUJOCO_KEY_MAX_LIN", "2.0"))
    max_yaw = float(os.environ.get("MUJOCO_KEY_MAX_YAW", "2.0"))
    env._keyboard_command_active = False

    def print_command():
        print(
            "[keyboard] cmd "
            f"x={env._ref_base_lin_vel_H[0]:.2f} m/s, "
            f"y={env._ref_base_lin_vel_H[1]:.2f} m/s, "
            f"yaw={env._ref_base_ang_yaw_dot:.2f} rad/s"
        )

    def clamp_command():
        env._ref_base_lin_vel_H[0] = np.clip(env._ref_base_lin_vel_H[0], -max_lin, max_lin)
        env._ref_base_lin_vel_H[1] = np.clip(env._ref_base_lin_vel_H[1], -max_lin, max_lin)
        env._ref_base_ang_yaw_dot = np.clip(env._ref_base_ang_yaw_dot, -max_yaw, max_yaw)

    def key_callback(keycode):
        handled = True
        if keycode in (KEY_W, KEY_UP):
            env._ref_base_lin_vel_H[0] += lin_step
        elif keycode in (KEY_S, KEY_DOWN):
            env._ref_base_lin_vel_H[0] -= lin_step
        elif keycode == KEY_A:
            env._ref_base_lin_vel_H[1] += lin_step
        elif keycode == KEY_D:
            env._ref_base_lin_vel_H[1] -= lin_step
        elif keycode in (KEY_Q, KEY_LEFT):
            env._ref_base_ang_yaw_dot += yaw_sign * yaw_step
        elif keycode in (KEY_E, KEY_RIGHT):
            env._ref_base_ang_yaw_dot -= yaw_sign * yaw_step
        elif keycode in (KEY_0, KEY_R, KEY_LEFT_CTRL, KEY_RIGHT_CTRL):
            env._ref_base_lin_vel_H[:] = 0.0
            env._ref_base_ang_yaw_dot = 0.0
        elif keycode == KEY_SPACE and env.viewer is not None:
            print("Pausing simulation." if not env.is_paused else "Resuming simulation.")
            env.is_paused = not env.is_paused
        else:
            handled = False

        if handled and keycode != KEY_SPACE:
            env._keyboard_command_active = True
            clamp_command()
            print_command()

    env._key_callback = key_callback
    print(
        "[keyboard] W/Up forward, S/Down backward, A/D lateral, "
        "Q/E or Left/Right yaw, R/0/Ctrl stop, Space pause"
    )


def _get_feet_contacts(env):
    try:
        _, feet_contacts, feet_grf = env.feet_contact_state(ground_reaction_forces=True)
        if feet_contacts is not None:
            return np.array(
                [
                    float(feet_contacts.FL),
                    float(feet_contacts.FR),
                    float(feet_contacts.RL),
                    float(feet_contacts.RR),
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                float(np.linalg.norm(feet_grf.FL) > 1.0),
                float(np.linalg.norm(feet_grf.FR) > 1.0),
                float(np.linalg.norm(feet_grf.RL) > 1.0),
                float(np.linalg.norm(feet_grf.RR) > 1.0),
            ],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(4, dtype=np.float32)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    scene_name = os.environ.get("SCENE", "flat")
    headless = os.environ.get("MUJOCO_HEADLESS", "0") == "1"
    max_steps = int(os.environ.get("MUJOCO_MAX_STEPS", "0"))
    render_freq = float(os.environ.get("MUJOCO_RENDER_FREQ", "30"))
    cmd_ramp_time = max(0.0, float(os.environ.get("MUJOCO_CMD_RAMP_TIME", "0.0")))
    cmd_yaw_sign = float(os.environ.get("MUJOCO_YAW_SIGN", "-1.0"))

    # Go2 posture was trained with dt=0.005 and decimation=4.
    simulation_dt = float(os.environ.get("MUJOCO_SIM_DT", "0.005"))
    env = QuadrupedEnv(
        robot="go2",
        scene=scene_name,
        sim_dt=simulation_dt,
        base_vel_command_type="human",
    )
    env.reset(random=False)

    locomotion_policy = Go2PosturePolicyWrapper(env=env)

    q0 = locomotion_policy.default_joint_pos
    env.mjData.qpos[2] = locomotion_policy.initial_base_height
    env.mjData.qpos[env.legs_qpos_idx.FL] = q0[0:3]
    env.mjData.qpos[env.legs_qpos_idx.FR] = q0[3:6]
    env.mjData.qpos[env.legs_qpos_idx.RL] = q0[6:9]
    env.mjData.qpos[env.legs_qpos_idx.RR] = q0[9:12]
    env.mjData.qvel[:] = 0.0
    mujoco.mj_forward(env.mjModel, env.mjData)

    if not headless:
        install_keyboard_command_callback(env)
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    control_decimation = max(1, round(1.0 / (locomotion_policy.RL_FREQ * simulation_dt)))
    last_render_time = time.time()

    while True:
        step_start = time.time()

        if cmd_ramp_time > 0.0:
            cmd_scale = min(1.0, env.step_num * simulation_dt / cmd_ramp_time)
        else:
            cmd_scale = 1.0
        keyboard_command_active = bool(getattr(env, "_keyboard_command_active", False))
        if "MUJOCO_CMD_X" in os.environ and not keyboard_command_active:
            env._ref_base_lin_vel_H[0] = cmd_scale * float(os.environ["MUJOCO_CMD_X"])
        if "MUJOCO_CMD_Y" in os.environ and not keyboard_command_active:
            env._ref_base_lin_vel_H[1] = cmd_scale * float(os.environ["MUJOCO_CMD_Y"])
        if "MUJOCO_CMD_YAW" in os.environ and not keyboard_command_active:
            env._ref_base_ang_yaw_dot = cmd_yaw_sign * cmd_scale * float(os.environ["MUJOCO_CMD_YAW"])

        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame="base")
        base_ang_vel = env.base_ang_vel(frame="base")
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos

        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]

        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]

        try:
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel(frame="base")
        except TypeError:
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if env.step_num % control_decimation == 0:
            desired_joint_pos = locomotion_policy.compute_control(
                base_pos=base_pos,
                base_quat_wxyz=base_quat_wxyz,
                base_lin_vel=base_lin_vel,
                base_ang_vel=base_ang_vel,
                joints_pos=joints_pos,
                joints_vel=joints_vel,
                ref_base_lin_vel=ref_base_lin_vel,
                ref_base_ang_vel=ref_base_ang_vel,
                feet_contacts=_get_feet_contacts(env),
            )
        else:
            desired_joint_pos = locomotion_policy.desired_joint_pos

        tau = LegsAttr(*[np.zeros((1, int(env.mjModel.nu / 4))) for _ in range(4)])
        tau.FL = locomotion_policy.Kp_walking * (desired_joint_pos.FL - joints_pos.FL) - locomotion_policy.Kd_walking * joints_vel.FL
        tau.FR = locomotion_policy.Kp_walking * (desired_joint_pos.FR - joints_pos.FR) - locomotion_policy.Kd_walking * joints_vel.FR
        tau.RL = locomotion_policy.Kp_walking * (desired_joint_pos.RL - joints_pos.RL) - locomotion_policy.Kd_walking * joints_vel.RL
        tau.RR = locomotion_policy.Kp_walking * (desired_joint_pos.RR - joints_pos.RR) - locomotion_policy.Kd_walking * joints_vel.RR

        if np.isfinite(locomotion_policy.effort_limit):
            tau.FL = np.clip(tau.FL, -locomotion_policy.effort_limit, locomotion_policy.effort_limit)
            tau.FR = np.clip(tau.FR, -locomotion_policy.effort_limit, locomotion_policy.effort_limit)
            tau.RL = np.clip(tau.RL, -locomotion_policy.effort_limit, locomotion_policy.effort_limit)
            tau.RR = np.clip(tau.RR, -locomotion_policy.effort_limit, locomotion_policy.effort_limit)

        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL.reshape((3,))
        action[env.legs_tau_idx.FR] = tau.FR.reshape((3,))
        action[env.legs_tau_idx.RL] = tau.RL.reshape((3,))
        action[env.legs_tau_idx.RR] = tau.RR.reshape((3,))
        env.step(action=action)

        if max_steps > 0 and env.step_num >= max_steps:
            break

        loop_elapsed_time = time.time() - step_start
        if loop_elapsed_time < simulation_dt:
            time.sleep(simulation_dt - loop_elapsed_time)

        if not headless and (time.time() - last_render_time > 1.0 / render_freq or env.step_num == 1):
            env.render()
            last_render_time = time.time()

    if headless:
        print(
            "[go2_posture_mujoco] final "
            f"step={env.step_num} base_pos={env.base_pos} "
            f"base_euler_xyz={env.base_ori_euler_xyz}"
        )
    env.close()
