import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped

from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator
from unitree_go.msg import LowCmd, LowState


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Unitree low-state order: FR, FL, RR, RL.
# DLS/controller order: FL, FR, RL, RR.
DLS_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def _zeros(count):
    return [0.0] * count


class Go2UnitreeBridge(Node):
    def __init__(self):
        super().__init__("go2_unitree_bridge")

        self.declare_parameter("publish_lowcmd", False)
        self.declare_parameter("lowcmd_topic", "/lowcmd")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("pose_topic", "/utlidar/robot_pose")
        self.declare_parameter("timeout_sec", 0.25)
        self.declare_parameter("kp_scale", 1.0)
        self.declare_parameter("kd_scale", 1.0)
        self.declare_parameter("max_position_step", 0.08)

        self.publish_lowcmd = self.get_parameter("publish_lowcmd").value
        self.lowcmd_topic = self.get_parameter("lowcmd_topic").value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.kp_scale = float(self.get_parameter("kp_scale").value)
        self.kd_scale = float(self.get_parameter("kd_scale").value)
        self.max_position_step = float(self.get_parameter("max_position_step").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.lowstate = None
        self.pose = None
        self.last_traj_time = 0.0
        self.latest_cmd = self._make_idle_lowcmd()
        self.last_commanded_position = None

        self.base_state_pub = self.create_publisher(BaseState, "/base_state", 1)
        self.blind_state_pub = self.create_publisher(BlindState, "blind_state", 1)
        self.imu_pub = self.create_publisher(Imu, "imu", 1)

        self.create_subscription(
            LowState,
            self.get_parameter("lowstate_topic").value,
            self.lowstate_cb,
            sensor_qos,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("pose_topic").value,
            self.pose_cb,
            sensor_qos,
        )
        self.create_subscription(
            TrajectoryGenerator,
            "/trajectory_generator",
            self.trajectory_cb,
            1,
        )

        self.lowcmd_pub = self.create_publisher(LowCmd, self.lowcmd_topic, command_qos)
        self.create_timer(0.002, self.timer_cb)

        mode = "ENABLED" if self.publish_lowcmd else "DRY-RUN"
        self.get_logger().warn(f"Go2 bridge started. lowcmd publishing: {mode}")
        self.get_logger().warn(
            f"Command safety: kp_scale={self.kp_scale}, kd_scale={self.kd_scale}, "
            f"max_position_step={self.max_position_step} rad/tick"
        )
        if not self.publish_lowcmd:
            self.get_logger().warn("Use --ros-args -p publish_lowcmd:=true only when the robot is safely supported.")

    def lowstate_cb(self, msg):
        self.lowstate = msg
        now = time.time()
        measured_positions = [float(msg.motor_state[i].q) for i in DLS_TO_UNITREE]
        if self.last_commanded_position is None:
            self.last_commanded_position = measured_positions.copy()

        blind = BlindState()
        blind.frame_id = "go2"
        blind.sequence_id = int(msg.tick)
        blind.timestamp = now
        blind.robot_name = "go2"
        blind.joints_name = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ]
        blind.joints_position = measured_positions
        blind.joints_velocity = [float(msg.motor_state[i].dq) for i in DLS_TO_UNITREE]
        blind.joints_effort = [float(msg.motor_state[i].tau_est) for i in DLS_TO_UNITREE]
        blind.joints_acceleration = _zeros(12)
        blind.joints_temperature = [float(msg.motor_state[i].temperature) for i in DLS_TO_UNITREE]
        blind.feet_contact = [bool(f > 0) for f in msg.foot_force]
        blind.current_feet_positions = _zeros(12)
        self.blind_state_pub.publish(blind)

        imu = Imu()
        imu.frame_id = "go2"
        imu.sequence_id = int(msg.tick)
        imu.timestamp = now
        imu.orientation = [float(x) for x in msg.imu_state.quaternion]
        imu.orientation_rpy = [float(x) for x in msg.imu_state.rpy]
        imu.orientation_covariance = _zeros(9)
        imu.angular_velocity = [float(x) for x in msg.imu_state.gyroscope]
        imu.angular_velocity_covariance = _zeros(9)
        imu.linear_acceleration = [float(x) for x in msg.imu_state.accelerometer]
        imu.linear_acceleration_covariance = _zeros(9)
        self.imu_pub.publish(imu)

        if self.pose is not None:
            base = BaseState()
            base.frame_id = "go2"
            base.sequence_id = int(msg.tick)
            base.timestamp = now
            base.robot_name = "go2"
            base.pose.position = [
                float(self.pose.pose.position.x),
                float(self.pose.pose.position.y),
                float(self.pose.pose.position.z),
            ]
            # DLS message uses xyzw for /base_state; run_controller rolls it to wxyz.
            base.pose.orientation = [
                float(self.pose.pose.orientation.x),
                float(self.pose.pose.orientation.y),
                float(self.pose.pose.orientation.z),
                float(self.pose.pose.orientation.w),
            ]
            base.velocity.linear = _zeros(3)
            base.velocity.angular = [float(x) for x in msg.imu_state.gyroscope]
            base.acceleration.linear = [float(x) for x in msg.imu_state.accelerometer]
            base.acceleration.angular = _zeros(3)
            base.stance_status = [bool(f > 0) for f in msg.foot_force]
            self.base_state_pub.publish(base)

    def pose_cb(self, msg):
        self.pose = msg

    def trajectory_cb(self, msg):
        cmd = self._make_idle_lowcmd()
        positions = list(msg.joints_position)
        velocities = list(msg.joints_velocity) if msg.joints_velocity else _zeros(12)
        kp = list(msg.kp) if msg.kp else _zeros(12)
        kd = list(msg.kd) if msg.kd else _zeros(12)
        tau = list(msg.joints_effort) if msg.joints_effort else _zeros(12)
        if self.last_commanded_position is None:
            self.last_commanded_position = positions.copy()

        limited_positions = []
        for previous, target in zip(self.last_commanded_position, positions):
            delta = max(-self.max_position_step, min(self.max_position_step, float(target) - float(previous)))
            limited_positions.append(float(previous) + delta)
        self.last_commanded_position = limited_positions.copy()

        for dls_idx, unitree_idx in enumerate(DLS_TO_UNITREE):
            motor = cmd.motor_cmd[unitree_idx]
            motor.mode = 0x01
            motor.q = float(limited_positions[dls_idx])
            motor.dq = float(velocities[dls_idx])
            motor.kp = float(kp[dls_idx]) * self.kp_scale
            motor.kd = float(kd[dls_idx]) * self.kd_scale
            motor.tau = float(tau[dls_idx])

        cmd.crc = self._lowcmd_crc(cmd)
        self.latest_cmd = cmd
        self.last_traj_time = time.time()

    def timer_cb(self):
        if not self.publish_lowcmd:
            return
        if time.time() - self.last_traj_time > self.timeout_sec:
            cmd = self._make_idle_lowcmd()
            cmd.crc = self._lowcmd_crc(cmd)
            self.lowcmd_pub.publish(cmd)
            return
        self.lowcmd_pub.publish(self.latest_cmd)

    def _make_idle_lowcmd(self):
        cmd = LowCmd()
        cmd.head = [0xFE, 0xEF]
        cmd.level_flag = 0xFF
        cmd.frame_reserve = 0
        cmd.sn = [0, 0]
        cmd.version = [0, 0]
        cmd.bandwidth = 0
        cmd.wireless_remote = [0] * 40
        cmd.led = [0] * 12
        cmd.fan = [0] * 2
        cmd.gpio = 0
        cmd.reserve = 0
        cmd.crc = 0
        for motor in cmd.motor_cmd:
            motor.mode = 0x01
            motor.q = POS_STOP_F
            motor.dq = VEL_STOP_F
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        cmd.crc = self._lowcmd_crc(cmd)
        return cmd

    def _lowcmd_crc(self, cmd):
        data = bytearray()
        data.extend(struct.pack(
            "<BBBBIIIIHxx",
            int(cmd.head[0]),
            int(cmd.head[1]),
            int(cmd.level_flag),
            int(cmd.frame_reserve),
            int(cmd.sn[0]),
            int(cmd.sn[1]),
            int(cmd.version[0]),
            int(cmd.version[1]),
            int(cmd.bandwidth),
        ))

        for motor in cmd.motor_cmd:
            data.extend(struct.pack(
                "<BxxxfffffIII",
                int(motor.mode),
                float(motor.q),
                float(motor.dq),
                float(motor.tau),
                float(motor.kp),
                float(motor.kd),
                int(motor.reserve[0]),
                int(motor.reserve[1]),
                int(motor.reserve[2]),
            ))

        data.extend(struct.pack(
            "<BBBB",
            int(cmd.bms_cmd.off),
            int(cmd.bms_cmd.reserve[0]),
            int(cmd.bms_cmd.reserve[1]),
            int(cmd.bms_cmd.reserve[2]),
        ))
        data.extend(bytes(int(x) & 0xFF for x in cmd.wireless_remote))
        data.extend(bytes(int(x) & 0xFF for x in cmd.led))
        data.extend(bytes(int(x) & 0xFF for x in cmd.fan))
        data.extend(struct.pack("<BxI", int(cmd.gpio), int(cmd.reserve)))

        if len(data) != 808:
            raise RuntimeError(f"Unexpected LowCmd CRC payload size: {len(data)}")

        crc = 0xFFFFFFFF
        polynomial = 0x04C11DB7
        for (word,) in struct.iter_unpack("<I", data):
            xbit = 1 << 31
            for _ in range(32):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ polynomial) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
                if word & xbit:
                    crc ^= polynomial
                xbit >>= 1
        return crc & 0xFFFFFFFF


def main():
    rclpy.init()
    node = Go2UnitreeBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
