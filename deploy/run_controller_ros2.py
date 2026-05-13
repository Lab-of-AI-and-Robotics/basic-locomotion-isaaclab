# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))



import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator

import time
import numpy as np
import torch
np.set_printoptions(precision=3, suppress=True)

import threading

import copy

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr


# Locomotion Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper
from go2_posture_policy_wrapper import Go2PosturePolicyWrapper

import config

# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_MUJOCO_RENDER = False


class ControllerROS2(Node):
    def __init__(self):
        super().__init__('ControllerROS2')

        self.declare_parameter("pose_topic", "/utlidar/robot_pose")
        self.declare_parameter("odom_topic", "/utlidar/robot_odom")
        self.declare_parameter("base_velocity_smoothing", 0.6)
        self.declare_parameter("base_velocity_zero_warmup", 1.0)
        self.declare_parameter("allow_zero_base_fallback", True)
        self.declare_parameter("rl_activation_blend_time", 2.0)
        self.declare_parameter("joystick_filter", 0.7)
        self.declare_parameter("joy_max_forward_velocity", 2.0)
        self.declare_parameter("joy_max_lateral_velocity", 1.0)
        self.declare_parameter("joy_max_yaw_rate", 1.2)
        self.declare_parameter("joy_yaw_sign", -1.0)
        # Source priority for base linear velocity: /base_state (HAL fusion) > leg odometry > lidar diff > zero.
        self.declare_parameter("base_state_freshness_sec", 0.1)
        self.declare_parameter("base_lin_vel_clip", 2.0)
        self.declare_parameter("base_lin_vel_lpf_tau", 0.05)
        self.declare_parameter("command_zero_threshold", 0.03)
        self.declare_parameter("joystick_timeout_sec", 1.0)
        self.declare_parameter("go2_posture_hold_initial_when_command_zero", True)
        self.declare_parameter("go2_posture_hold_initial_on_joy_timeout", True)
        self.declare_parameter("go2_posture_hold_initial_use_stand_gains", True)
        self.declare_parameter("go2_posture_hold_max_joint_step", 0.05)
        self.declare_parameter("sport_mode_l2_button_index", 5)
        self.declare_parameter("sport_mode_a_button_index", 8)
        self.declare_parameter("sport_mode_toggle_debounce_sec", 0.5)
        self.declare_parameter("kill_button_index", -1)
        self.base_velocity_smoothing = max(0.0, min(1.0, float(self.get_parameter("base_velocity_smoothing").value)))
        self.base_velocity_zero_warmup = max(0.0, float(self.get_parameter("base_velocity_zero_warmup").value))
        self.allow_zero_base_fallback = bool(self.get_parameter("allow_zero_base_fallback").value)
        self.rl_activation_blend_time = max(0.0, float(self.get_parameter("rl_activation_blend_time").value))
        self.joystick_filter = max(0.0, min(1.0, float(self.get_parameter("joystick_filter").value)))
        self.joy_max_forward_velocity = max(0.0, float(self.get_parameter("joy_max_forward_velocity").value))
        self.joy_max_lateral_velocity = max(0.0, float(self.get_parameter("joy_max_lateral_velocity").value))
        self.joy_max_yaw_rate = max(0.0, float(self.get_parameter("joy_max_yaw_rate").value))
        self.joy_yaw_sign = float(self.get_parameter("joy_yaw_sign").value)
        self.base_state_freshness_sec = max(0.01, float(self.get_parameter("base_state_freshness_sec").value))
        self.base_lin_vel_clip = max(0.1, float(self.get_parameter("base_lin_vel_clip").value))
        self.base_lin_vel_lpf_tau = max(0.0, float(self.get_parameter("base_lin_vel_lpf_tau").value))
        self.command_zero_threshold = max(
            0.0, float(self.get_parameter("command_zero_threshold").value)
        )
        self.joystick_timeout_sec = max(0.0, float(self.get_parameter("joystick_timeout_sec").value))
        self.go2_posture_hold_initial_when_command_zero = bool(
            self.get_parameter("go2_posture_hold_initial_when_command_zero").value
        )
        self.go2_posture_hold_initial_on_joy_timeout = bool(
            self.get_parameter("go2_posture_hold_initial_on_joy_timeout").value
        )
        self.go2_posture_hold_initial_use_stand_gains = bool(
            self.get_parameter("go2_posture_hold_initial_use_stand_gains").value
        )
        self.go2_posture_hold_max_joint_step = max(
            0.0, float(self.get_parameter("go2_posture_hold_max_joint_step").value)
        )
        self.sport_mode_l2_button_index = int(self.get_parameter("sport_mode_l2_button_index").value)
        self.sport_mode_a_button_index = int(self.get_parameter("sport_mode_a_button_index").value)
        self.sport_mode_toggle_debounce_sec = max(
            0.0, float(self.get_parameter("sport_mode_toggle_debounce_sec").value)
        )
        self.kill_button_index = int(self.get_parameter("kill_button_index").value)

        # Mujoco env
        robot_name = config.robot
        scene_name = config.scene
        self.active_env_cfg = config.active_training_env()
        simulation_dt = float(self.active_env_cfg["sim"]["dt"])
        self.policy_backend = config.policy_backend


        # Create the quadruped robot environment -----------------------------------------------------------
        self.env = QuadrupedEnv(
            robot=robot_name,
            scene=scene_name,
            sim_dt=simulation_dt,
            base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
        )
        self.env.reset(random=False)
        
        self.last_render_time = time.time()
        if USE_MUJOCO_RENDER:
            self.env.render()   
                 

        # Subscribers and Publishers
        self.subscription_base_state = self.create_subscription(BaseState,"/base_state", self.get_base_state_callback, 1)
        self.subscription_robot_pose = self.create_subscription(
            PoseStamped, self.get_parameter("pose_topic").value, self.get_robot_pose_callback, 1
        )
        self.subscription_robot_odom = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.get_robot_odom_callback, 1
        )
        self.subscription_blind_state = self.create_subscription(BlindState,"blind_state", self.get_blind_state_callback, 1)
        self.subscription_imu = self.create_subscription(Imu,"imu", self.get_imu_callback, 1)
        
        self.subscription_joy = self.create_subscription(Joy,"joy", self.get_joy_callback, 1)
        self.last_joy_time = None
        
        self.publisher_trajectory_generator = self.create_publisher(TrajectoryGenerator,"/trajectory_generator", 1)
        self.sequence_id = 0 # To keep track of the last msg sent, useful for debugging and synchronization


        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False 
        self.first_message_imu_arrived = False
        self.was_rl_activated = False
        self.rl_activation_time = None
        self.rl_activation_start_joint_pos = None
        self.last_missing_base_warning_time = 0.0
        self.zero_base_fallback_warned = False
        self._sport_mode_combo_was_pressed = False
        self._last_sport_mode_toggle_time = 0.0
        self.joystick_timed_out = False
        self.go2_posture_initial_hold_active = False
        self._last_desired_joint_pos = None

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.previous_pose_position = None
        self.previous_pose_time = None
        # Tracks freshness/source of base linear-velocity estimate.
        self.last_base_state_time = None
        self.last_lidar_velocity_time = None
        self._linear_velocity_source = "none"
        self._warned_stale_base_state = False
        self._warned_lidar_velocity_fallback = False
        # Leg-odometry buffers: HAL does not publish /base_state, so we run
        # our own kinematic-inertial filter from IMU + joint state.
        self._last_r_foot_b = None
        self._last_fk_time = None

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.feet_contacts = np.zeros(4)

        # IMU
        self.imu_linear_acceleration = np.zeros(3)
        self.imu_angular_velocity = np.zeros(3)
        self.imu_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        
        # Initialization of variables used in the main control loop --------------------------------
        if config.policy_backend == "go2_posture":
            self.locomotion_policy = Go2PosturePolicyWrapper(
                env=self.env,
                run_dir=config.go2_posture_run_dir,
                checkpoint=config.go2_posture_checkpoint,
                device=config.go2_posture_device,
                use_exported_adaptation=True,
            )
            self.get_logger().warn(
                "Using go2_posture policy with exported concurrent-SE/RMA networks for deploy."
            )
        elif config.policy_backend == "basic":
            self.locomotion_policy = LocomotionPolicyWrapper(env=self.env)
        else:
            raise ValueError(f"Unsupported policy_backend={config.policy_backend}")

        self.timer = self.create_timer(1.0/self.locomotion_policy.RL_FREQ, self.compute_rl_control)


        self.stand_up_and_down_actions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        keyframe_id = mujoco.mj_name2id(self.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.env.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions.FL = goDown_qpos[7:10]
        self.stand_up_and_down_actions.FR = goDown_qpos[10:13]
        self.stand_up_and_down_actions.RL = goDown_qpos[13:16]
        self.stand_up_and_down_actions.RR = goDown_qpos[16:19]
        self.joint_positions = goDown_qpos[7:19]


        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

    
    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a 
        8Bitdi Ultimate 2C Wireless Controller.
        """

        filter_joystick = self.joystick_filter
        axis_0 = msg.axes[0] if len(msg.axes) > 0 else 0.0
        axis_1 = msg.axes[1] if len(msg.axes) > 1 else 0.0
        axis_3 = msg.axes[3] if len(msg.axes) > 3 else 0.0
        target_x = np.clip(axis_1, -1.0, 1.0) * self.joy_max_forward_velocity
        target_y = np.clip(axis_0, -1.0, 1.0) * self.joy_max_lateral_velocity
        target_yaw = self.joy_yaw_sign * np.clip(axis_3, -1.0, 1.0) * self.joy_max_yaw_rate
        self.env._ref_base_lin_vel_H[0] = self.env._ref_base_lin_vel_H[0]*filter_joystick + target_x*(1-filter_joystick)  # Forward/Backward
        self.env._ref_base_lin_vel_H[1] = self.env._ref_base_lin_vel_H[1]*filter_joystick + target_y*(1-filter_joystick)  # Left/Right
        self.env._ref_base_ang_yaw_dot = self.env._ref_base_ang_yaw_dot*filter_joystick + target_yaw*(1-filter_joystick)  # Yaw

        self.last_joy_time = time.time()
        self.joystick_timed_out = False

        sport_combo_pressed = (
            self._joy_button_pressed(msg, self.sport_mode_l2_button_index)
            and self._joy_button_pressed(msg, self.sport_mode_a_button_index)
        )
        now = time.time()
        if (
            sport_combo_pressed
            and not self._sport_mode_combo_was_pressed
            and now - self._last_sport_mode_toggle_time >= self.sport_mode_toggle_debounce_sec
            and hasattr(self, "console")
        ):
            self.console.isRLActivated = not self.console.isRLActivated
            self._last_sport_mode_toggle_time = now
            if self.console.isRLActivated:
                self.get_logger().warn("Joystick L2+A: sport mode ON, RL policy activated.")
            else:
                self.get_logger().warn("Joystick L2+A: sport mode OFF, RL policy deactivated.")
        self._sport_mode_combo_was_pressed = sport_combo_pressed

        if self._joy_button_pressed(msg, self.kill_button_index):
            self.get_logger().info("Joystick button pressed, shutting down the node.") 
            # This will kill the robot hal
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            # This will kill the process running this script
            os.system("pkill -f play_ros2.py") 
            exit(0)


    def _joy_button_pressed(self, msg, button_index):
        return 0 <= button_index < len(msg.buttons) and msg.buttons[button_index] == 1


    def get_base_state_callback(self, msg):
        # /base_state is the primary source: HAL already fuses Unitree IMU
        # with leg kinematics. Trust it for both pose and twist.
        self.position = np.array(msg.pose.position) #world frame
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.orientation = np.roll(np.array(msg.pose.orientation), 1) #world frame
        self.linear_velocity = np.array(msg.velocity.linear) #world frame
        self.angular_velocity = np.array(msg.velocity.angular) #base frame
        self.last_base_state_time = time.time()
        self._linear_velocity_source = "base_state"
        self._warned_stale_base_state = False

        self.first_message_base_arrived = True


    def get_robot_pose_callback(self, msg):
        # Lidar SLAM pose: keep position for visualization. Only touch
        # orientation / linear velocity if HAL /base_state is unavailable.
        self.position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        if self._base_state_is_stale():
            self.orientation = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ])
            self._update_base_linear_velocity_from_pose(self.position, self._stamp_to_sec(msg.header.stamp))
        self.first_message_base_arrived = True


    def get_robot_odom_callback(self, msg):
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        if self._base_state_is_stale():
            self.orientation = np.array([
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ])
            self.angular_velocity = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ])
            self._update_base_linear_velocity_from_pose(self.position, self._stamp_to_sec(msg.header.stamp))
        self.first_message_base_arrived = True


    def _base_state_is_stale(self):
        if self.last_base_state_time is None:
            return True
        return (time.time() - self.last_base_state_time) > self.base_state_freshness_sec


    def _policy_uses_imu_observation(self):
        return (
            self.policy_backend == "go2_posture"
            or self.active_env_cfg.get("use_imu", False)
            or self.active_env_cfg.get("use_concurrent_state_est", False)
            or self.active_env_cfg.get("use_concurrent_state_estimator", False)
        )


    def _estimate_base_linear_velocity(self):
        """Return the world-frame base linear velocity for the policy.

        Priority: HAL /base_state -> leg odometry -> lidar diff -> zero.
        Clip + 1st-order LPF protect the policy from out-of-distribution spikes.
        """
        now = time.time()
        if not self._base_state_is_stale():
            v = self.linear_velocity
        else:
            v_leg = self._leg_odometry_velocity()
            if v_leg is not None:
                v = v_leg
                if not self._warned_stale_base_state:
                    self.get_logger().info("Using leg odometry for base linear velocity.")
                    self._warned_stale_base_state = True
            elif self.last_lidar_velocity_time is not None and (now - self.last_lidar_velocity_time) < 0.5:
                v = self.linear_velocity
                if not self._warned_stale_base_state:
                    self.get_logger().warn(
                        "Falling back to lidar pose differential for base linear velocity."
                    )
                    self._warned_stale_base_state = True
            else:
                v = np.zeros(3)
                if not self._warned_stale_base_state:
                    self.get_logger().warn("No fresh base velocity source; feeding zero to policy.")
                    self._warned_stale_base_state = True

        v = np.clip(v, -self.base_lin_vel_clip, self.base_lin_vel_clip)

        if self.base_lin_vel_lpf_tau > 0.0:
            dt = max(1e-3, float(self.loop_time))
            alpha = dt / (self.base_lin_vel_lpf_tau + dt)
            if not hasattr(self, "_base_lin_vel_filtered"):
                self._base_lin_vel_filtered = v.copy()
            self._base_lin_vel_filtered = (1.0 - alpha) * self._base_lin_vel_filtered + alpha * v
            v = self._base_lin_vel_filtered.copy()
        return v


    def _leg_odometry_velocity(self):
        """Estimate world-frame base linear velocity from IMU + joint state.

        Assumes the two feet with the lowest base-frame z are in contact and
        have zero world velocity (no slip). Returns None until two FK samples
        have accumulated, or if joint state is unavailable.
        """
        if not self.first_message_joints_arrived or not self.first_message_imu_arrived:
            return None

        q = np.asarray(self.joint_positions, dtype=np.float32).reshape(-1)
        if q.size != 12:
            return None

        # joint_positions arrives leg-grouped: [FL_h, FL_t, FL_c, FR_h, ..., RR_c].
        # Go2Solver expects solver-grouped: [hip x4, thigh x4, calf x4].
        q_legs = q.reshape(4, 3)
        q_solver = np.concatenate([q_legs[:, 0], q_legs[:, 1], q_legs[:, 2]]).astype(np.float32)

        if not hasattr(self.locomotion_policy, "solver"):
            return None

        device = self.locomotion_policy.device if hasattr(self.locomotion_policy, "device") else torch.device("cpu")
        with torch.no_grad():
            q_t = torch.tensor(q_solver, dtype=torch.float32, device=device).view(1, 12)
            r_foot_b = self.locomotion_policy.solver.go2_fk_new(q_t).view(4, 3).detach().cpu().numpy()

        now = time.time()
        if self._last_r_foot_b is None or self._last_fk_time is None:
            self._last_r_foot_b = r_foot_b.copy()
            self._last_fk_time = now
            return None

        dt = now - self._last_fk_time
        if dt < 1e-3:
            return None
        rdot_foot_b = (r_foot_b - self._last_r_foot_b) / dt
        self._last_r_foot_b = r_foot_b.copy()
        self._last_fk_time = now

        omega_b = np.asarray(self.imu_angular_velocity, dtype=np.float32).reshape(3)
        # v_base_b = -(omega_b x r_foot_b + d/dt r_foot_b)  per contacting foot
        v_base_b_per_foot = -(np.cross(np.broadcast_to(omega_b, r_foot_b.shape), r_foot_b) + rdot_foot_b)

        # Contact selection: two feet with lowest (most negative) base-frame z.
        z_foot_b = r_foot_b[:, 2]
        contact_idx = np.argsort(z_foot_b)[:2]
        v_base_b = v_base_b_per_foot[contact_idx].mean(axis=0)

        R_b = self._quat_to_rotmat(self.imu_orientation)
        v_base_w = R_b @ v_base_b
        if not np.isfinite(v_base_w).all():
            return None
        return v_base_w.astype(np.float32)


    @staticmethod
    def _quat_to_rotmat(q_wxyz):
        q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1.0e-8:
            return np.eye(3, dtype=np.float32)
        w, x, y, z = q / n
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float32)


    def _update_base_linear_velocity_from_pose(self, position, stamp_time):
        current_time = stamp_time if stamp_time > 0.0 else time.time()
        if self.previous_pose_position is not None and self.previous_pose_time is not None:
            dt = current_time - self.previous_pose_time
            if dt > 1e-4:
                raw_velocity = (position - self.previous_pose_position) / dt
                smoothing = self.base_velocity_smoothing
                self.linear_velocity = smoothing * self.linear_velocity + (1.0 - smoothing) * raw_velocity
                self.last_lidar_velocity_time = current_time
                self._linear_velocity_source = "lidar_diff"
                if not self._warned_lidar_velocity_fallback:
                    self.get_logger().warn(
                        "Falling back to lidar pose differential for base linear velocity "
                        "(HAL /base_state unavailable or stale)."
                    )
                    self._warned_lidar_velocity_fallback = True

        self.previous_pose_position = position.copy()
        self.previous_pose_time = current_time


    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9



    def get_blind_state_callback(self, msg):
        self.joint_positions = np.array(msg.joints_position)
        self.joint_velocities = np.array(msg.joints_velocity)
        if len(msg.feet_contact) >= 4:
            self.feet_contacts = np.array(msg.feet_contact[:4], dtype=np.float32)

        self.first_message_joints_arrived = True
     
        
    def get_imu_callback(self, msg):
        self.imu_linear_acceleration = np.array(msg.linear_acceleration) 
        self.imu_angular_velocity = np.array(msg.angular_velocity) 
        # For the quaternion, the order is [x, y, z, w] on DLS2 but here we want [w, x, y, z] (mujoco convention)
        self.imu_orientation = np.roll(np.array(msg.orientation), 1) 

        self.first_message_imu_arrived = True


    def compute_rl_control(self):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time
        

        # Stand-up/down only needs joint state. RL walking needs the state source
        # matching the selected policy configuration.
        if(self.first_message_joints_arrived==False):
            return
        if(self.console.isRLActivated):
            if(self._policy_uses_imu_observation()):
                if(self.first_message_imu_arrived==False):
                    return
            elif(self.first_message_base_arrived==False and not (self.allow_zero_base_fallback and self.first_message_imu_arrived)):
                now = time.time()
                if(now - self.last_missing_base_warning_time > 1.0):
                    self.get_logger().warn("RL active but no /base_state, /utlidar/robot_pose, or /utlidar/robot_odom received yet.")
                    self.last_missing_base_warning_time = now
                return

        if(self.console.isRLActivated and not self.was_rl_activated):
            self.rl_activation_time = time.time()
            self.rl_activation_start_joint_pos = self._legs_attr_from_flat_joints(self.joint_positions)
            self.linear_velocity = np.zeros(3)
            self.previous_pose_position = None
            self.previous_pose_time = None
            self.get_logger().warn(
                f"RL activated. Holding base linear velocity at zero for {self.base_velocity_zero_warmup:.2f}s."
            )
        self.was_rl_activated = self.console.isRLActivated

        base_linear_velocity = self._estimate_base_linear_velocity()
        if(self.console.isRLActivated and self.rl_activation_time is not None):
            if(time.time() - self.rl_activation_time < self.base_velocity_zero_warmup):
                base_linear_velocity = np.zeros(3)

        use_zero_base_fallback = (
            self.console.isRLActivated
            and not self.first_message_base_arrived
            and self.allow_zero_base_fallback
            and self.first_message_imu_arrived
            and not self._policy_uses_imu_observation()
        )
        if(use_zero_base_fallback and not self.zero_base_fallback_warned):
            self.get_logger().warn("Using IMU orientation and zero linear velocity because no base pose/odom source is available.")
            self.zero_base_fallback_warned = True
        
        # Update the mujoco model
        # Note that in case of IMU or concurrent state estimator, these info below are not used,
        # In the case we have a state estimator, this is usefull only for debugging visually
        self.env.mjData.qpos[0:3] = copy.deepcopy(self.position)
        self.env.mjData.qvel[0:3] = copy.deepcopy(base_linear_velocity)

        if(self._policy_uses_imu_observation()):
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.imu_orientation)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.imu_angular_velocity)
        elif(use_zero_base_fallback):
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.imu_orientation)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.imu_angular_velocity)
        else:
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
        
        # These info instead are used for sure in all the cases
        self.env.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
        self.env.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
        self.env.mjModel.opt.timestep = simulation_dt
        mujoco.mj_forward(self.env.mjModel, self.env.mjData) 
        
        # Safety check for joystick timeout
        if(
            self.last_joy_time is not None
            and self.joystick_timeout_sec > 0.0
            and time.time() - self.last_joy_time > self.joystick_timeout_sec
        ):
            self.env._ref_base_lin_vel_H[0] = 0.0
            self.env._ref_base_lin_vel_H[1] = 0.0
            self.env._ref_base_ang_yaw_dot = 0.0
            print("Joystick timeout, stopping the robot")
            self.last_joy_time = None
            self.joystick_timed_out = True

        self.env._ref_base_lin_vel_H[0] = np.clip(
            self.env._ref_base_lin_vel_H[0],
            -self.joy_max_forward_velocity,
            self.joy_max_forward_velocity,
        )
        self.env._ref_base_lin_vel_H[1] = np.clip(
            self.env._ref_base_lin_vel_H[1],
            -self.joy_max_lateral_velocity,
            self.joy_max_lateral_velocity,
        )
        self.env._ref_base_ang_yaw_dot = np.clip(
            self.env._ref_base_ang_yaw_dot,
            -self.joy_max_yaw_rate,
            self.joy_max_yaw_rate,
        )
            

        env = self.env
        locomotion_policy = self.locomotion_policy
        
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame='base')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos


        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]

        # variable saved for goDown and goUp motion
        self.joint_positions = np.concatenate([joints_pos.FL, joints_pos.FR, joints_pos.RL, joints_pos.RR], axis=0).flatten()
    
        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        if config.policy_backend == "go2_posture":
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel(frame='base')
        else:
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        go2_posture_hold_initial = self._go2_posture_should_hold_initial(ref_base_lin_vel, ref_base_ang_vel)
        if(self.console.isRLActivated):

            if config.policy_backend == "go2_posture" and go2_posture_hold_initial:
                desired_joint_pos = self._go2_posture_initial_target(locomotion_policy)
                if self.go2_posture_hold_max_joint_step > 0.0:
                    desired_joint_pos = self._limit_leg_target_step(
                        desired_joint_pos, self.go2_posture_hold_max_joint_step
                    )
            elif config.policy_backend == "go2_posture":
                desired_joint_pos = locomotion_policy.compute_control(
                        base_pos=base_pos,
                        base_quat_wxyz=base_quat_wxyz,
                        base_lin_vel=base_lin_vel,
                        base_ang_vel=base_ang_vel,
                        joints_pos=joints_pos,
                        joints_vel=joints_vel,
                        ref_base_lin_vel=ref_base_lin_vel,
                        ref_base_ang_vel=ref_base_ang_vel,
                        feet_contacts=self.feet_contacts)
            else:
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
                        imu_linear_acceleration=self.imu_linear_acceleration,
                        imu_angular_velocity=self.imu_angular_velocity,
                        imu_orientation=self.imu_orientation)

            if(self.rl_activation_time is not None and self.rl_activation_blend_time > 0.0):
                blend_alpha = min(1.0, (time.time() - self.rl_activation_time) / self.rl_activation_blend_time)
                desired_joint_pos = self._blend_leg_targets(
                    self.rl_activation_start_joint_pos,
                    desired_joint_pos,
                    blend_alpha,
                )
            
            # Impedence Loop
            if go2_posture_hold_initial and self.go2_posture_hold_initial_use_stand_gains:
                Kp = locomotion_policy.Kp_stand_up_and_down
                Kd = locomotion_policy.Kd_stand_up_and_down
            else:
                Kp = locomotion_policy.Kp_walking
                Kd = locomotion_policy.Kd_walking


        else:
            desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
            desired_joint_pos.FL = self.stand_up_and_down_actions.FL
            desired_joint_pos.FR = self.stand_up_and_down_actions.FR
            desired_joint_pos.RL = self.stand_up_and_down_actions.RL
            desired_joint_pos.RR = self.stand_up_and_down_actions.RR

            # Impedence Loop
            Kp = locomotion_policy.Kp_stand_up_and_down
            Kd = locomotion_policy.Kd_stand_up_and_down

        # Publish the desired joint positions to the trajectory generator --------------------------------
        trajectory_generator_msg = TrajectoryGenerator()
        trajectory_generator_msg.timestamp = float(self.get_clock().now().nanoseconds)
        trajectory_generator_msg.sequence_id = int(self.sequence_id % 1000)  # To avoid overflow, we reset the sequence id after it reaches a certain value
        self.sequence_id += 1
        trajectory_generator_msg.joints_position = np.array([desired_joint_pos.FL, desired_joint_pos.FR, desired_joint_pos.RL, desired_joint_pos.RR]).flatten().tolist()
        trajectory_generator_msg.joints_velocity = np.zeros(12).tolist()
        trajectory_generator_msg.kp = (np.ones(12) * Kp).tolist()
        trajectory_generator_msg.kd = (np.ones(12) * Kd).tolist()
        self._last_desired_joint_pos = self._flat_joints_from_legs_attr(desired_joint_pos)

        self.publisher_trajectory_generator.publish(trajectory_generator_msg)
        
        
        
        # Render the simulation at a certain frequency -----------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30  # Hz
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ or self.env.step_num == 1:
                self.env.render()
                self.last_render_time = time.time()


    def _legs_attr_from_flat_joints(self, joints):
        legs = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        joints = np.array(joints).flatten()
        legs.FL = joints[0:3]
        legs.FR = joints[3:6]
        legs.RL = joints[6:9]
        legs.RR = joints[9:12]
        return legs

    def _go2_posture_should_hold_initial(self, ref_base_lin_vel, ref_base_ang_vel):
        if config.policy_backend != "go2_posture" or not self.console.isRLActivated:
            self.go2_posture_initial_hold_active = False
            return False
        if not self.go2_posture_hold_initial_when_command_zero:
            self.go2_posture_initial_hold_active = False
            return False

        command_norm = max(
            abs(float(ref_base_lin_vel[0])),
            abs(float(ref_base_lin_vel[1])),
            abs(float(ref_base_ang_vel[2])),
        )
        command_is_zero = command_norm < self.command_zero_threshold
        hold = command_is_zero or (
            self.go2_posture_hold_initial_on_joy_timeout and self.joystick_timed_out
        )
        if hold != self.go2_posture_initial_hold_active:
            if hold:
                self.get_logger().warn(
                    "Go2 posture holding initial pose because command is zero or joystick timed out."
                )
            else:
                self.get_logger().warn("Go2 posture leaving initial-pose hold; policy command is active.")
        self.go2_posture_initial_hold_active = hold
        return hold


    def _go2_posture_initial_target(self, locomotion_policy):
        if hasattr(locomotion_policy, "default_joint_pos"):
            return self._legs_attr_from_flat_joints(locomotion_policy.default_joint_pos)
        return self._legs_attr_from_flat_joints(self.joint_positions)


    def _flat_joints_from_legs_attr(self, joints):
        return np.concatenate(
            [
                np.asarray(joints.FL).reshape(-1),
                np.asarray(joints.FR).reshape(-1),
                np.asarray(joints.RL).reshape(-1),
                np.asarray(joints.RR).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)


    def _limit_leg_target_step(self, target, max_step):
        if self._last_desired_joint_pos is None:
            return target
        target_flat = self._flat_joints_from_legs_attr(target)
        previous = np.asarray(self._last_desired_joint_pos, dtype=np.float32).reshape(12)
        limited = previous + np.clip(target_flat - previous, -max_step, max_step)
        return self._legs_attr_from_flat_joints(limited)


    def _blend_leg_targets(self, start, target, alpha):
        if(start is None):
            return target
        blended = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        blended.FL = (1.0 - alpha) * start.FL + alpha * target.FL
        blended.FR = (1.0 - alpha) * start.FR + alpha * target.FR
        blended.RL = (1.0 - alpha) * start.RL + alpha * target.RL
        blended.RR = (1.0 - alpha) * start.RR + alpha * target.RR
        return blended




#---------------------------
if __name__ == '__main__':
    
    print('Hello from basic-locomotion-dls-isaaclab ros node.')
    
    rclpy.init()
    controller_ros2_node = ControllerROS2()
    rclpy.spin(controller_ros2_node)
    
    controller_ros2_node.destroy_node()
    rclpy.shutdown()

    print("ControllerROS2 node is stopped")
    exit(0)
