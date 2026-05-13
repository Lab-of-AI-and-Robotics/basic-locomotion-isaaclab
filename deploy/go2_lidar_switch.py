#!/usr/bin/env python3
"""Switch Unitree Go2 lidar-related services via /api/robot_state.

This talks to the Unitree RobotState API. It does not kill local ROS nodes; it
asks the robot to switch the matching onboard service off or on.
"""

import argparse
import json
import re
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node

from unitree_api.msg import Request, Response


ROBOT_STATE_API_ID_SERVICE_SWITCH = 1001
ROBOT_STATE_API_ID_SERVICE_LIST = 1003

DEFAULT_LIDAR_PATTERN = r"(lidar|utlidar|slam)"
LIDAR_CONSUMER_SERVICES = [
    "obstacles_avoid",
    "voxel_height_mapping",
    "unitree_lidar_slam",
    "unitree_lidar",
]


class RobotStateServiceSwitch(Node):
    def __init__(self, timeout_sec, publish_period_sec):
        super().__init__("go2_lidar_switch")
        self.timeout_sec = float(timeout_sec)
        self.publish_period_sec = float(publish_period_sec)
        self._responses = defaultdict(list)
        self._pub = self.create_publisher(Request, "/api/robot_state/request", 10)
        self._sub = self.create_subscription(
            Response,
            "/api/robot_state/response",
            self._response_callback,
            10,
        )

    def _response_callback(self, msg):
        self._responses[int(msg.header.identity.api_id)].append(msg)

    @staticmethod
    def _request_id():
        return time.monotonic_ns()

    def call(self, api_id, parameter=None):
        req = Request()
        req.header.identity.api_id = int(api_id)
        req.header.identity.id = self._request_id()
        if parameter is not None:
            req.parameter = json.dumps(parameter)

        api_id = int(req.header.identity.api_id)
        self._responses[api_id].clear()
        deadline = time.time() + self.timeout_sec
        next_publish_time = 0.0
        while rclpy.ok() and time.time() < deadline:
            now = time.time()
            if now >= next_publish_time:
                self._pub.publish(req)
                next_publish_time = now + self.publish_period_sec
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._responses[api_id]:
                msg = self._responses[api_id].pop(0)
                if msg.data:
                    return json.loads(msg.data)
                return None

        raise TimeoutError(
            f"Timed out waiting for /api/robot_state response api_id={api_id}. "
            "Check robot network, ros2_connect.bash, and Unitree DDS."
        )

    def service_list(self):
        data = self.call(ROBOT_STATE_API_ID_SERVICE_LIST)
        if data is None:
            return []
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected service list response: {data!r}")
        return data

    def service_switch(self, service_name, enabled):
        data = self.call(
            ROBOT_STATE_API_ID_SERVICE_SWITCH,
            {"name": service_name, "switch": int(bool(enabled))},
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected switch response for {service_name}: {data!r}")
        return data


def _matching_services(services, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    matches = []
    for item in services:
        name = str(item.get("name", ""))
        if regex.search(name):
            matches.append(name)
    return matches


def _print_services(services, names=None):
    name_filter = set(names) if names is not None else None
    for item in services:
        name = str(item.get("name", ""))
        if name_filter is not None and name not in name_filter:
            continue
        print(
            f"  name={name} "
            f"status={item.get('status')} protect={item.get('protect')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Turn Unitree Go2 lidar-related robot services off or on."
    )
    parser.add_argument(
        "--on",
        action="store_true",
        help="Switch matched services on instead of off.",
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="Exact robot service name to switch. Can be repeated.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_LIDAR_PATTERN,
        help=f"Regex used to auto-select services when --service is omitted. Default: {DEFAULT_LIDAR_PATTERN}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print robot services; do not switch anything.",
    )
    parser.add_argument(
        "--include-consumers",
        action="store_true",
        help=(
            "Also switch common lidar consumer services. For off, this switches "
            "obstacles_avoid and voxel_height_mapping before lidar/slam."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="Response timeout for each RobotState API call.",
    )
    parser.add_argument(
        "--publish-period-sec",
        type=float,
        default=0.25,
        help="Re-publish period while waiting for a RobotState API response.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = RobotStateServiceSwitch(args.timeout_sec, args.publish_period_sec)
    try:
        services = node.service_list()
        if args.list:
            print("Robot services:")
            _print_services(services)
            return

        enabled = bool(args.on)
        targets = list(args.service)
        if args.include_consumers:
            known_names = {str(item.get("name", "")) for item in services}
            if enabled:
                ordered = list(reversed(LIDAR_CONSUMER_SERVICES))
            else:
                ordered = list(LIDAR_CONSUMER_SERVICES)
            targets.extend(name for name in ordered if name in known_names)
        if not targets:
            targets = _matching_services(services, args.pattern)
        targets = list(dict.fromkeys(targets))

        if not targets:
            raise SystemExit(
                "No lidar-like service matched. Re-run with --list, then pass "
                "the exact name using --service <name>."
            )

        action = "ON" if enabled else "OFF"
        for name in targets:
            result = node.service_switch(name, enabled)
            print(f"Switched {name} {action}: {result}")

        refreshed = node.service_list()
        print("Selected service status after switch:")
        _print_services(refreshed, targets)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
