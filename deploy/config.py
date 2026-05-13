import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = os.environ.get("ROBOT", "go2")  # 'aliengo', 'go2', 'spot', 'b2', 'hyqreal2'
scene = os.environ.get("SCENE", "flat")  # flat, random_boxes, random_pyramids, perlin

policy_backend = os.environ.get("POLICY_BACKEND", "go2_posture")  # "basic" or "go2_posture"
go2_posture_policy_root = os.environ.get(
    "GO2_POSTURE_POLICY_ROOT",
    os.path.abspath(os.path.join(dir_path, "..", "tested_policies", "go2", "go2_posture_guidance")),
)
go2_posture_run_name = os.environ.get(
    "GO2_POSTURE_RUN_NAME",
    "2026-05-04_23-04-13_postureON_clampON_air0.0",
)
go2_posture_run_dir = os.environ.get(
    "GO2_POSTURE_RUN_DIR",
    os.path.join(go2_posture_policy_root, go2_posture_run_name),
)
go2_posture_checkpoint = os.environ.get(
    "GO2_POSTURE_CHECKPOINT",
    os.path.join(go2_posture_run_dir, "model_9999.pt"),
)
go2_posture_device = os.environ.get("GO2_POSTURE_DEVICE", "cpu")
go2_posture_env_cfg = None

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

elif(robot == "spot"):
    Kp_walking = 25.0
    Kd_walking = 0.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

    # Keep a valid fallback path so legacy basic-policy config loading remains harmless.
    policy_folder_path = dir_path + "/../tested_policies/go2/symmetricactor_data_augmented"

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
training_env = {}
training_env_path = policy_folder_path + "/params/env.yaml"
if os.path.isfile(training_env_path):
    with open(training_env_path, "r") as file:
        training_env = yaml.unsafe_load(file)

go2_posture_env_cfg_path = os.path.join(go2_posture_run_dir, "params", "env.yaml")
if os.path.isfile(go2_posture_env_cfg_path):
    with open(go2_posture_env_cfg_path, "r") as file:
        go2_posture_env_cfg = yaml.unsafe_load(file)

def active_training_env():
    if policy_backend == "go2_posture" and go2_posture_env_cfg is not None:
        return go2_posture_env_cfg
    return training_env
