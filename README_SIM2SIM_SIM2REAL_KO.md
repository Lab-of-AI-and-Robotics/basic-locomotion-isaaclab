# Sim-to-Sim / Sim-to-Real 실행 가이드

이 문서는 `basic-locomotion-isaaclab`에서 학습된 Go2 정책을 PC 환경에서 실행하는 절차를 정리한다. Jetson/JetPack 로컬 실행 내용은 포함하지 않는다.

## 1. 기본 구조

이 프로젝트의 deploy 실행은 크게 두 가지로 나뉜다.

```text
Sim-to-Sim
  RL controller -> MuJoCo simulator

Sim-to-Real
  RL controller -> unitree-ros2-dls HAL -> Unitree Go2
```

ROS2 실기 구동에서는 다음 토픽들이 핵심이다.

```text
/lowstate              Go2 -> HAL
/blind_state           HAL -> controller
/imu                   HAL -> controller
/trajectory_generator  controller -> HAL
/lowcmd                HAL -> Go2
```

Unitree 기본 조종기를 사용할 때는 다음 변환 브리지를 추가로 사용한다.

```text
/wirelesscontroller    Go2 기본 조종기 토픽
/joy                   controller가 읽는 표준 joystick 토픽
```

## 2. 환경 준비

deploy용 conda 환경을 활성화한다.

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
```

ROS2 환경을 사용할 때는 각 터미널에서 반드시 해당 실행 방식에 맞는 connect script를 source한다.

Sim-to-Sim ROS2:

```bash
source deploy/ros2_localhost_connect.sh
```

Sim-to-Real Go2:

```bash
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
```

주의: 같은 터미널에서 아래 workspace를 섞어 source하지 않는다.

```bash
source /opt/ros/humble/setup.bash
source ~/go2_ws/install/setup.bash
source deploy/go2_direct_connect.sh
```

Python ABI가 섞이면 `rclpy._rclpy_pybind11` 또는 `UnsupportedTypeSupport` 오류가 날 수 있다.

## 3. 정책 설정

정책과 게인은 `deploy/config.py`에서 선택한다.

Go2 설정 예:

```python
robot = 'go2'

elif(robot == "go2"):
    Kp_walking = 25.0
    Kd_walking = 0.8

    Kp_stand_up_and_down = 25.0
    Kd_stand_up_and_down = 0.8

    policy_folder_path = dir_path + "/../tested_policies/" + robot + "/concurrent_se"
```

Go2에서 주로 확인한 정책 폴더는 다음과 같다.

```text
tested_policies/go2/symmetricactor_data_augmented
tested_policies/go2/concurrent_se
tested_policies/go2/concurrent_se_24k
tested_policies/go2/heightmap_frontal_and_se
```

`symmetricactor_data_augmented`는 `use_imu: false`, `use_concurrent_state_est: false` 설정이라 `/base_state`가 필요하다. 별도 state estimator 없이 `unitree-ros2-dls` HAL만 사용할 경우 `/base_state` publisher가 없어서 controller가 실제 command를 publish하지 않을 수 있다.

`concurrent_se`는 `/imu`와 `/blind_state` 기반으로 동작할 수 있어, 현재 Go2 실기 테스트에서는 이 정책을 사용했다.

정책 설정 확인:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
python3 - <<'PY'
import deploy.config as config
print(config.policy_folder_path)
print("use_imu:", config.training_env["use_imu"])
print("use_concurrent_state_est:", config.training_env["use_concurrent_state_est"])
PY
```

## 4. Sim-to-Sim 실행

### 4.1 MuJoCo 단독 실행

가장 단순한 sim-to-sim 실행이다.

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
python3 deploy/play_mujoco.py
```

### 4.2 ROS2 기반 Sim-to-Sim

ROS2 controller와 simulator를 분리해서 실행한다.

터미널 1, controller:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source deploy/ros2_localhost_connect.sh
python3 deploy/run_controller_ros2.py
```

터미널 2, simulator:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source deploy/ros2_localhost_connect.sh
python3 deploy/run_simulator_ros2.py
```

controller 콘솔에서 다음 순서로 실행한다.

```text
goUp
activate
ictp
```

키보드 제어:

```text
w/s  전진/후진
a/d  좌/우 이동
q/e  yaw 회전
0    속도 명령 정지
```

## 5. Sim-to-Real 실행

### 5.1 네트워크 확인

PC와 Go2를 랜선으로 연결한다. PC의 유선 인터페이스에 Go2와 같은 대역의 정적 IP를 설정한다.

예:

```text
PC:  192.168.123.10/24
Go2: 192.168.123.161
```

확인:

```bash
ip -brief addr
ping -c 3 192.168.123.161
```

`unitree-ros2-dls/ros2_connect.bash`의 CycloneDDS 인터페이스 이름이 실제 유선 인터페이스와 맞아야 한다.

예:

```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name="enp13s0" priority="default" multicast="default" />
</Interfaces></General></Domain></CycloneDDS>'
```

Go2 DDS 토픽 확인:

```bash
cd ~/isaac_projects/unitree-ros2-dls
conda activate basic_locomotion_isaaclab_ros2_env
source ros2_connect.bash
ros2 topic list | grep lowstate
```

정상이라면 `/lowstate` 또는 `/lf/lowstate`가 보인다.

### 5.2 unitree-ros2-dls 빌드

`unitree-ros2-dls`는 Go2와 ROS2 controller 사이의 HAL 역할을 한다.

```bash
cd ~/isaac_projects
git clone https://github.com/iit-DLSLab/unitree-ros2-dls.git
cd unitree-ros2-dls
git submodule update --init unitree_ros2
```

Z1 arm 관련 submodule은 Go2 보행 deploy에 필요하지 않다.

빌드:

```bash
cd ~/isaac_projects/unitree-ros2-dls/unitree_ros2/cyclonedds_ws
conda activate basic_locomotion_isaaclab_ros2_env
colcon build

cd ~/isaac_projects/unitree-ros2-dls
source unitree_ros2/cyclonedds_ws/install/setup.bash
cd ros2_ws
colcon build
```

### 5.3 실기 실행 순서

터미널 1, Unitree HAL:

```bash
cd ~/isaac_projects/unitree-ros2-dls
conda activate basic_locomotion_isaaclab_ros2_env
source ros2_connect.bash
python3 launch_quadruped_hal.py
```

터미널 2, RL controller:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
python3 deploy/run_controller_ros2.py
```

토픽 연결 확인:

```bash
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
ros2 topic info -v /trajectory_generator
ros2 topic info -v /blind_state
ros2 topic info -v /imu
ros2 topic info -v /lowcmd
```

정상 상태:

```text
/trajectory_generator  publisher: ControllerROS2, subscriber: low_level_cmd_node
/blind_state           publisher: low_level_cmd_node, subscriber: ControllerROS2
/imu                   publisher: low_level_cmd_node, subscriber: ControllerROS2
/lowcmd                publisher: low_level_cmd_node, subscriber: Go2 DDS app
```

controller 콘솔에서:

```text
goUp
activate
ictp
```

키보드 제어는 sim-to-sim과 동일하다.

## 6. Unitree 기본 조종기 사용

Unitree 기본 조종기는 PC의 `/dev/input/js0`로 잡히는 일반 joystick이 아니다. Go2가 `/wirelesscontroller` 토픽으로 publish한다.

확인:

```bash
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
ros2 topic type /wirelesscontroller
ros2 topic info /wirelesscontroller
```

타입:

```text
unitree_go/msg/WirelessController
```

현재 controller는 `sensor_msgs/msg/Joy` 타입의 `/joy`를 읽으므로 변환 브리지를 실행한다.

터미널 3, Unitree 조종기 bridge:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
python3 deploy/unitree_wireless_to_joy.py --ros-args -p invert_lx:=true
```

`invert_lx:=true`는 좌우 이동 방향을 뒤집기 위한 설정이다. 방향이 다르면 다음 옵션을 조정한다.

```text
좌우 이동 반대:  -p invert_lx:=true
전후 이동 반대:  -p invert_ly:=true
yaw 회전 반대:   -p invert_rx:=true
```

확인:

```bash
ros2 topic echo /joy
```

스틱을 움직였을 때 `/joy.axes` 값이 변하면 정상이다.

조종기를 사용할 때는 controller 콘솔에서 다음만 실행하면 된다.

```text
goUp
activate
```

`ictp`는 키보드 제어 모드라 조종기 사용 시 필수는 아니다.

## 7. 안전 종료 순서

실기에서 `Ctrl+C`를 먼저 누르면 `/lowcmd`가 끊기면서 로봇이 주저앉을 수 있다. 종료는 반드시 controller 콘솔에서 자세를 낮춘 뒤 진행한다.

권장 순서:

```text
activate   # RL 비활성화
goDown     # 로봇을 앉히거나 눕힘
```

로봇이 완전히 내려간 것을 확인한 뒤 각 터미널에서 종료한다.

```text
controller 터미널: Ctrl+C
wireless bridge 터미널: Ctrl+C
HAL 터미널: Ctrl+C
```

비상 상황에서는 조이스틱/키보드 속도 명령을 0으로 만든 뒤 즉시 controller와 HAL을 종료하고, Unitree 앱/기본 조종기에서 damping 또는 safe 상태로 전환한다.

## 8. 자주 보는 문제

### rclpy._rclpy_pybind11 오류

Python 3.12 conda env와 `/opt/ros/humble` Python 3.10 패키지가 섞였을 때 발생한다.

해결:

```bash
conda activate basic_locomotion_isaaclab_ros2_env
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
python3 -c "import rclpy; print(rclpy.__file__)"
```

정상 경로:

```text
/home/lair0/anaconda3/envs/basic_locomotion_isaaclab_ros2_env/lib/python3.12/site-packages/rclpy/__init__.py
```

### /trajectory_generator가 publish되지 않음

정책이 `/base_state`를 요구하지만 `/base_state` publisher가 없으면 controller가 내부 safety check에서 return한다.

확인:

```bash
ros2 topic info -v /base_state
```

해결:

```text
1. state estimator를 실행해서 /base_state를 제공한다.
2. 또는 Go2 policy를 concurrent_se 계열로 바꾼다.
```

### HAL은 켜졌는데 로봇이 안 움직임

다음을 확인한다.

```bash
ros2 topic info -v /trajectory_generator
ros2 topic echo --once /trajectory_generator
ros2 topic echo --once /lowcmd
```

`/trajectory_generator`에 subscriber가 없으면 HAL이 연결되지 않은 것이다. `/lowcmd`의 `kp`, `kd`, `q` 값이 갱신되지 않고 `PosStopF` 형태로 남아 있으면 controller command가 HAL까지 오지 않은 것이다.

### 조종기 장치가 joy_enumerate_devices에 안 보임

Unitree 기본 조종기는 PC joystick 장치가 아니라 `/wirelesscontroller` 토픽으로 들어온다. `joy_node`가 아니라 `deploy/unitree_wireless_to_joy.py`를 사용한다.

일반 USB/Bluetooth 게임패드를 쓸 때만 다음 명령이 의미가 있다.

```bash
ros2 run joy joy_enumerate_devices
ros2 run joy joy_node --ros-args -p device_id:=0 -p deadzone:=0.08 -p autorepeat_rate:=20.0
```

## 9. 빠른 실행 요약

Go2 실기, Unitree 기본 조종기 사용:

터미널 1:

```bash
cd ~/isaac_projects/unitree-ros2-dls
conda activate basic_locomotion_isaaclab_ros2_env
source ros2_connect.bash
python3 launch_quadruped_hal.py
```

터미널 2:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
python3 deploy/run_controller_ros2.py
```

터미널 3:

```bash
cd ~/isaac_projects/basic-locomotion-isaaclab
conda activate basic_locomotion_isaaclab_ros2_env
source ~/isaac_projects/unitree-ros2-dls/ros2_connect.bash
python3 deploy/unitree_wireless_to_joy.py --ros-args -p invert_lx:=true
```

controller 콘솔:

```text
goUp
activate
```

종료:

```text
activate
goDown
Ctrl+C
```
