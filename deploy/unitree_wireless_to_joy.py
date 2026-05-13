#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from unitree_go.msg import WirelessController


BUTTON_BITS = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}


class UnitreeWirelessToJoy(Node):
    def __init__(self):
        super().__init__("unitree_wireless_to_joy")

        self.declare_parameter("deadzone", 0.05)
        self.declare_parameter("invert_lx", False)
        self.declare_parameter("invert_ly", False)
        self.declare_parameter("invert_rx", False)
        self.declare_parameter("invert_ry", False)

        self.deadzone = float(self.get_parameter("deadzone").value)
        self.invert_lx = bool(self.get_parameter("invert_lx").value)
        self.invert_ly = bool(self.get_parameter("invert_ly").value)
        self.invert_rx = bool(self.get_parameter("invert_rx").value)
        self.invert_ry = bool(self.get_parameter("invert_ry").value)

        self.pub = self.create_publisher(Joy, "/joy", 1)
        self.sub = self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self.wireless_callback,
            1,
        )

        self.get_logger().info(
            "Bridging /wirelesscontroller to /joy with Unitree button bit mapping"
        )

    def clean_axis(self, value, invert):
        if not math.isfinite(value):
            return 0.0
        if abs(value) < self.deadzone:
            return 0.0
        return float(-value if invert else value)

    def wireless_callback(self, msg):
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()

        lx = self.clean_axis(msg.lx, self.invert_lx)
        ly = self.clean_axis(msg.ly, self.invert_ly)
        rx = self.clean_axis(msg.rx, self.invert_rx)
        ry = self.clean_axis(msg.ry, self.invert_ry)

        # run_controller_ros2.py uses axes[1] for forward/backward,
        # axes[0] for lateral motion, and axes[3] for yaw.
        joy.axes = [lx, ly, ry, rx]

        keys = int(msg.keys)
        joy.buttons = [0] * 16
        for bit in BUTTON_BITS.values():
            joy.buttons[bit] = 1 if keys & (1 << bit) else 0

        self.pub.publish(joy)


def main():
    rclpy.init()
    node = UnitreeWirelessToJoy()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
