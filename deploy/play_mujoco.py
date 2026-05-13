# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi

import time
import numpy as np
from tqdm import tqdm
import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.mujoco.visual import render_sphere

# Locomotion Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config


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


def _parse_float_list_env(name, expected_len):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated floats")
    return np.asarray(values, dtype=np.float64)


def apply_mujoco_contact_overrides(env):
    floor_friction = _parse_float_list_env("MUJOCO_FLOOR_FRICTION", 3)
    foot_friction = _parse_float_list_env("MUJOCO_FOOT_FRICTION", 3)
    foot_solimp = _parse_float_list_env("MUJOCO_FOOT_SOLIMP", 5)
    foot_solref = _parse_float_list_env("MUJOCO_FOOT_SOLREF", 2)
    foot_condim = os.environ.get("MUJOCO_FOOT_CONDIM", "").strip()

    if floor_friction is not None:
        floor_id = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            env.mjModel.geom_friction[floor_id] = floor_friction
            print(f"[mujoco] floor friction override: {floor_friction}")

    foot_ids = [
        mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("FL", "FR", "RL", "RR")
    ]
    foot_ids = [idx for idx in foot_ids if idx >= 0]
    if not foot_ids:
        return

    if foot_friction is not None:
        for idx in foot_ids:
            env.mjModel.geom_friction[idx] = foot_friction
        print(f"[mujoco] foot friction override: {foot_friction}")
    if foot_solimp is not None:
        for idx in foot_ids:
            env.mjModel.geom_solimp[idx] = foot_solimp
        print(f"[mujoco] foot solimp override: {foot_solimp}")
    if foot_solref is not None:
        for idx in foot_ids:
            env.mjModel.geom_solref[idx] = foot_solref
        print(f"[mujoco] foot solref override: {foot_solref}")
    if foot_condim:
        condim = int(foot_condim)
        for idx in foot_ids:
            env.mjModel.geom_condim[idx] = condim
        print(f"[mujoco] foot condim override: {condim}")


def install_keyboard_command_callback(env):
    lin_step = float(os.environ.get("MUJOCO_KEY_LIN_STEP", "0.1"))
    yaw_step = float(os.environ.get("MUJOCO_KEY_YAW_STEP", "0.2"))
    yaw_sign = float(os.environ.get("MUJOCO_YAW_SIGN", "-1.0"))
    max_lin = float(os.environ.get("MUJOCO_KEY_MAX_LIN", "1.0"))
    max_yaw = float(os.environ.get("MUJOCO_KEY_MAX_YAW", "1.2"))
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
            print('Pausing simulation.' if not env.is_paused else 'Resuming simulation.')
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


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    scene_name = config.scene
    active_env_cfg = config.active_training_env()
    simulation_dt = float(active_env_cfg["sim"]["dt"])


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    )
    apply_mujoco_contact_overrides(env)


    env.reset(random=False)

    # Initialization of variables used in the main control loop --------------------------------
    if config.policy_backend == "basic":
        locomotion_policy = LocomotionPolicyWrapper(env=env)
    else:
        raise ValueError(f"Unsupported policy_backend={config.policy_backend}")

    headless = os.environ.get("MUJOCO_HEADLESS", "0") == "1"
    max_steps = int(os.environ.get("MUJOCO_MAX_STEPS", "0"))
    cmd_ramp_time = max(0.0, float(os.environ.get("MUJOCO_CMD_RAMP_TIME", "0.0")))
    cmd_yaw_sign = float(os.environ.get("MUJOCO_YAW_SIGN", "-1.0"))
    if not headless:
        install_keyboard_command_callback(env)
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False



    if(locomotion_policy.use_vision):
        resolution_heightmap = active_env_cfg["height_scanner2"]["pattern_cfg"]["resolution"]
        num_rows_heightmap = round(active_env_cfg["height_scanner2"]["pattern_cfg"]["size"][0]/resolution_heightmap) + 1
        num_cols_heightmap = round(active_env_cfg["height_scanner2"]["pattern_cfg"]["size"][1]/resolution_heightmap) + 1
        heightmap_offset = active_env_cfg["height_scanner2"]["offset"]
        heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData)     
    

    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    last_render_time = time.time()

    while True:
        step_start = time.time()
        
        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame='base')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos

        if(
            active_env_cfg.get("use_imu", False)
            or active_env_cfg.get("use_concurrent_state_est", False)
        ):
            sensordata = np.asarray(env.mjData.sensordata, dtype=np.float32)
            imu_linear_acceleration = sensordata[0:3] if sensordata.size >= 3 else np.zeros(3)
            imu_angular_velocity = sensordata[3:6] if sensordata.size >= 6 else base_ang_vel
            imu_orientation = base_quat_wxyz
        else:
            imu_linear_acceleration = np.zeros(3)
            imu_angular_velocity = np.zeros(3)
            imu_orientation = np.zeros(4)

        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]
    
        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        cmd_scale = 1.0
        if cmd_ramp_time > 0.0:
            cmd_scale = min(1.0, env.step_num * simulation_dt / cmd_ramp_time)
        keyboard_command_active = bool(getattr(env, "_keyboard_command_active", False))
        if "MUJOCO_CMD_X" in os.environ and not keyboard_command_active:
            env._ref_base_lin_vel_H[0] = cmd_scale * float(os.environ["MUJOCO_CMD_X"])
        if "MUJOCO_CMD_Y" in os.environ and not keyboard_command_active:
            env._ref_base_lin_vel_H[1] = cmd_scale * float(os.environ["MUJOCO_CMD_Y"])
        if "MUJOCO_CMD_YAW" in os.environ and not keyboard_command_active:
            env._ref_base_ang_yaw_dot = cmd_yaw_sign * cmd_scale * float(os.environ["MUJOCO_CMD_YAW"])
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if(locomotion_policy.use_vision):
            offset_world_frame = heightmap_offset["pos"] @ heading_orientation_SO3.T
            heightmap.update_height_map(env.mjData.qpos[0:3] + offset_world_frame, yaw=env.base_ori_euler_xyz[2])

        # RL controller --------------------------------------------------------------
        if env.step_num % round(1 / (locomotion_policy.RL_FREQ * simulation_dt)) == 0:            
            
            desired_joint_pos = locomotion_policy.compute_control(
                        base_pos=base_pos, 
                        base_ori_euler_xyz=base_ori_euler_xyz, 
                        base_quat_wxyz=base_quat_wxyz,
                        base_lin_vel=base_lin_vel, 
                        base_ang_vel=base_ang_vel,
                        heading_orientation_SO3=heading_orientation_SO3,
                        joints_pos=joints_pos, 
                        joints_vel=joints_vel,
                        ref_base_lin_vel=ref_base_lin_vel, 
                        ref_base_ang_vel=ref_base_ang_vel,
                        imu_linear_acceleration=imu_linear_acceleration,
                        imu_angular_velocity=imu_angular_velocity,
                        imu_orientation=imu_orientation,
                        heightmap_data=heightmap.data if locomotion_policy.use_vision else None)

        # PD controller --------------------------------------------------------------
        else:
            desired_joint_pos = locomotion_policy.desired_joint_pos


        Kp = locomotion_policy.Kp_walking
        Kd = locomotion_policy.Kd_walking

        error_joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        error_joints_pos.FL = desired_joint_pos.FL - joints_pos.FL
        error_joints_pos.FR = desired_joint_pos.FR - joints_pos.FR
        error_joints_pos.RL = desired_joint_pos.RL - joints_pos.RL
        error_joints_pos.RR = desired_joint_pos.RR - joints_pos.RR
        
        tau = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        tau.FL = Kp * (error_joints_pos.FL) - Kd * joints_vel.FL
        tau.FR = Kp * (error_joints_pos.FR) - Kd * joints_vel.FR
        tau.RL = Kp * (error_joints_pos.RL) - Kd * joints_vel.RL
        tau.RR = Kp * (error_joints_pos.RR) - Kd * joints_vel.RR
        effort_limit = getattr(locomotion_policy, "effort_limit", None)
        if effort_limit is not None:
            if np.isscalar(effort_limit):
                if np.isfinite(effort_limit):
                    tau.FL = np.clip(tau.FL, -effort_limit, effort_limit)
                    tau.FR = np.clip(tau.FR, -effort_limit, effort_limit)
                    tau.RL = np.clip(tau.RL, -effort_limit, effort_limit)
                    tau.RR = np.clip(tau.RR, -effort_limit, effort_limit)
            else:
                limits = np.asarray(effort_limit, dtype=np.float32).reshape(4, 3)
                tau.FL = np.clip(tau.FL, -limits[0], limits[0])
                tau.FR = np.clip(tau.FR, -limits[1], limits[1])
                tau.RL = np.clip(tau.RL, -limits[2], limits[2])
                tau.RR = np.clip(tau.RR, -limits[3], limits[3])

        if hasattr(locomotion_policy, "clip_effort_like_isaaclab_dc_motor"):
            tau_legs = np.stack(
                [
                    tau.FL.reshape(3),
                    tau.FR.reshape(3),
                    tau.RL.reshape(3),
                    tau.RR.reshape(3),
                ],
                axis=0,
            )
            qvel_legs = np.stack(
                [
                    joints_vel.FL.reshape(3),
                    joints_vel.FR.reshape(3),
                    joints_vel.RL.reshape(3),
                    joints_vel.RR.reshape(3),
                ],
                axis=0,
            )
            tau_legs = locomotion_policy.clip_effort_like_isaaclab_dc_motor(tau_legs, qvel_legs)
            tau.FL = tau_legs[0]
            tau.FR = tau_legs[1]
            tau.RL = tau_legs[2]
            tau.RR = tau_legs[3]


        # Set control and mujoco step ----------------------------------------------------------------------
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL.reshape((3,))
        action[env.legs_tau_idx.FR] = tau.FR.reshape((3,))
        action[env.legs_tau_idx.RL] = tau.RL.reshape((3,))
        action[env.legs_tau_idx.RR] = tau.RR.reshape((3,))
        state, reward, is_terminated, is_truncated, info = env.step(action=action)
        if max_steps > 0 and env.step_num >= max_steps:
            break


        # Sleep to match real-time ---------------------------------------------------------
        loop_elapsed_time = time.time() - step_start

        if(loop_elapsed_time < simulation_dt):
            time.sleep(simulation_dt - (loop_elapsed_time))

        # Render only at a certain frequency -----------------------------------------------------------------
        if not headless and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
            env.render()
            last_render_time = time.time()

            if(locomotion_policy.use_vision):
                if heightmap.data is not None:
                    for i in range(heightmap.data.shape[0]):
                        for j in range(heightmap.data.data.shape[1]):
                            heightmap.geom_ids[i, j] = render_sphere(
                                viewer=env.viewer,
                                position=([heightmap.data[i][j][0][0], heightmap.data[i][j][0][1], heightmap.data[i][j][0][2]]),
                                diameter=0.02,
                                color=[0, 1, 0, 0.5],
                                geom_id=heightmap.geom_ids[i, j],
                            )


    if headless:
        print(
            "[mujoco] final "
            f"step={env.step_num} base_pos={env.base_pos} "
            f"base_euler_xyz={env.base_ori_euler_xyz}"
        )

    if hasattr(locomotion_policy, "flush_mujoco_trace"):
        locomotion_policy.flush_mujoco_trace()

    env.close()
