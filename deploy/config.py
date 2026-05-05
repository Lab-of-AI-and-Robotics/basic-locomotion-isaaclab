import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = os.environ.get("ROBOT", "go2")  # 'aliengo', 'go2', 'b2', 'hyqreal2'
scene = os.environ.get("SCENE", "flat")  # flat, random_boxes, random_pyramids, perlin

policy_backend = os.environ.get("POLICY_BACKEND", "go2_posture")  # "basic", "safe_rl", or "go2_posture"
safe_rl_payload_path = os.environ.get(
    "SAFE_RL_PAYLOAD",
    os.path.abspath(os.path.join(dir_path, "..", "safe_rl_basic_locomotion_handoff", "payload")),
)
go2_posture_run_dir = os.environ.get(
    "GO2_POSTURE_RUN_DIR",
    "/home/lair0/isaaclab_ws/go2_posture/logs/rsl_rl/go2_posture_direct/"
    "2026-05-04_23-04-13_postureON_clampON_air0.0",
)
go2_posture_checkpoint = os.environ.get(
    "GO2_POSTURE_CHECKPOINT",
    os.path.join(go2_posture_run_dir, "model_9999.pt"),
)
go2_posture_device = os.environ.get("GO2_POSTURE_DEVICE", "cpu")
go2_posture_env_cfg = None

# Pick one safe_rl policy here. It can still be overridden from the terminal with
# SAFE_RL_POLICY_NAME=<name>.
safe_rl_available_policies = [
    "transfer_flat_l1_debug",
    "baseheight_020_support_nf",
    "baseheight_015_support_nf",
    "joint_rom_stand_fl_pct050",
    "joint_rom_stand_fl_pct020",
    "joint_rom_stand_fl_pct010",
    "joint_rom_stand_fl_pct000",
    "joint_rom_shiftx020_fl_pct050",
    "joint_rom_shiftx020_fl_pct020",
    "joint_rom_shiftx020_fl_pct010",
    "joint_rom_shiftx020_fl_pct000",
]
safe_rl_selected_policy = "joint_rom_stand_fl_pct050"
safe_rl_policy_name = os.environ.get("SAFE_RL_POLICY_NAME", safe_rl_selected_policy)
if safe_rl_policy_name not in safe_rl_available_policies:
    raise ValueError(
        f"Unknown safe_rl policy: {safe_rl_policy_name!r}. "
        f"Choose one of: {', '.join(safe_rl_available_policies)}"
    )
safe_rl_device = os.environ.get("SAFE_RL_DEVICE", "cpu")
safe_rl_zero_action_tolerance = 0.35
safe_rl_debug_steps = int(os.environ.get("SAFE_RL_DEBUG_STEPS", "10"))
safe_rl_env_cfg = None

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 21.5
    Kd_walking = 3.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

    policy_folder_path = dir_path + "/../tested_policies/" + robot + "/symmetricactor_data_augmented"

elif(robot == "go2"):
    Kp_walking = 20.0
    Kd_walking = 2.0 #1.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

    policy_folder_path = dir_path + "/../tested_policies/" + robot + "/symmetricactor_data_augmented"

elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.
elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.

    Kp_stand_up_and_down = 175.
    Kd_stand_up_and_down = 20.
else:
    raise ValueError(f"Robot {robot} not supported")

# ----------------------------------------------------------------------------------------------------------------

concurrent_state_est_network = policy_folder_path + "/exported/concurrent_state_estimator.pth"
rma_network = policy_folder_path + "/exported/rma.pth"

# Load specific training parameters
import yaml 
with open(policy_folder_path + "/params/env.yaml", "r") as file:
    training_env = yaml.unsafe_load(file)

safe_rl_env_cfg_path = os.path.join(safe_rl_payload_path, "policies", safe_rl_policy_name, "params", "env.yaml")
if os.path.isfile(safe_rl_env_cfg_path):
    with open(safe_rl_env_cfg_path, "r") as file:
        safe_rl_env_cfg = yaml.unsafe_load(file)

go2_posture_env_cfg_path = os.path.join(go2_posture_run_dir, "params", "env.yaml")
if os.path.isfile(go2_posture_env_cfg_path):
    with open(go2_posture_env_cfg_path, "r") as file:
        go2_posture_env_cfg = yaml.unsafe_load(file)

def active_training_env():
    if policy_backend == "go2_posture" and go2_posture_env_cfg is not None:
        return go2_posture_env_cfg
    if policy_backend == "safe_rl" and safe_rl_env_cfg is not None:
        return safe_rl_env_cfg
    return training_env
