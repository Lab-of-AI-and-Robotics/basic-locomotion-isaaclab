#!/usr/bin/env python3
"""Switch Go2 UTLiDAR rotation/data via the Unitree DDS switch topic.

Unitree's SDK2 example publishes std_msgs/String with "OFF" or "ON" to the DDS
topic rt/utlidar/switch. With ROS 2, publishing to /utlidar/switch maps to that
DDS topic when the Unitree CycloneDDS environment is sourced.
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class UTLidarSwitch(Node):
    def __init__(self, topic):
        super().__init__("go2_utlidar_spin_switch")
        self.publisher = self.create_publisher(String, topic, 10)

    def publish_command(self, command, duration_sec, period_sec):
        msg = String()
        msg.data = command

        deadline = time.time() + duration_sec
        while rclpy.ok() and time.time() < deadline:
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period_sec)


def main():
    parser = argparse.ArgumentParser(
        description="Publish ON/OFF to the Go2 UTLiDAR switch topic."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--on", action="store_true", help="Switch UTLiDAR on.")
    group.add_argument("--off", action="store_true", help="Switch UTLiDAR off.")
    parser.add_argument(
        "--topic",
        default="/utlidar/switch",
        help="ROS 2 topic mapped to DDS rt/utlidar/switch.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=2.0,
        help="How long to repeat the command.",
    )
    parser.add_argument(
        "--period-sec",
        type=float,
        default=0.1,
        help="Command publish period.",
    )
    args = parser.parse_args()

    command = "ON" if args.on else "OFF"

    rclpy.init()
    node = UTLidarSwitch(args.topic)
    try:
        node.get_logger().info(f"Publishing {command} to {args.topic}")
        node.publish_command(command, args.duration_sec, args.period_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
