import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from gym_quadruped.utils.quadruped_utils import LegsAttr


DEFAULT_POLICY_ROOT = Path(
    os.environ.get(
        "GO2_POSTURE_POLICY_ROOT",
        Path(__file__).resolve().parents[1] / "tested_policies" / "go2" / "go2_posture_guidance",
    )
).expanduser()
DEFAULT_RUN_NAME = os.environ.get(
    "GO2_POSTURE_RUN_NAME",
    "2026-05-04_23-04-13_postureON_clampON_air0.0",
)
DEFAULT_RUN_DIR = DEFAULT_POLICY_ROOT / DEFAULT_RUN_NAME


def _flat_legs(legs):
    return np.concatenate(
        [
            np.asarray(legs.FL).reshape(-1),
            np.asarray(legs.FR).reshape(-1),
            np.asarray(legs.RL).reshape(-1),
            np.asarray(legs.RR).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def _leg_grouped_to_joint_grouped(q):
    q = np.asarray(q, dtype=np.float32).reshape(4, 3)
    return np.concatenate([q[:, 0], q[:, 1], q[:, 2]], axis=0)


def _joint_grouped_to_leg_grouped(q):
    q = np.asarray(q, dtype=np.float32).reshape(3, 4)
    return q.T.reshape(12)


def _legs_attr_from_flat(q):
    q = np.asarray(q, dtype=np.float32).reshape(12)
    legs = LegsAttr(*[np.zeros((1, 3), dtype=np.float32) for _ in range(4)])
    legs.FL = q[0:3]
    legs.FR = q[3:6]
    legs.RL = q[6:9]
    legs.RR = q[9:12]
    return legs


class _ActorMLP(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        layer_ids = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("actor.") and key.endswith(".weight")
            }
        )
        modules = []
        for i, layer_id in enumerate(layer_ids):
            weight = state_dict[f"actor.{layer_id}.weight"]
            modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
            if i < len(layer_ids) - 1:
                modules.append(nn.ELU())
        self.actor = nn.Sequential(*modules)
        self.actor.load_state_dict(
            {
                key.removeprefix("actor."): value
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
        )
        self.input_dim = int(state_dict[f"actor.{layer_ids[0]}.weight"].shape[1])

    def forward(self, obs):
        return self.actor(obs)


class _PrefixedSequentialMLP(nn.Module):
    def __init__(self, state_dict, prefix):
        super().__init__()
        layer_ids = sorted(
            {
                int(key.removeprefix(prefix).split(".")[0])
                for key in state_dict
                if key.startswith(prefix) and key.endswith(".weight")
            }
        )
        if not layer_ids:
            raise ValueError(f"Cannot find MLP weights with prefix={prefix!r}")

        modules = []
        for i, layer_id in enumerate(layer_ids):
            weight = state_dict[f"{prefix}{layer_id}.weight"]
            modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
            if i < len(layer_ids) - 1:
                modules.append(nn.ELU())
        self.net = nn.Sequential(*modules)
        self.net.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        )
        first_weight = state_dict[f"{prefix}{layer_ids[0]}.weight"]
        last_weight = state_dict[f"{prefix}{layer_ids[-1]}.weight"]
        self.input_dim = int(first_weight.shape[1])
        self.output_dim = int(last_weight.shape[0])

    def forward(self, inputs):
        return self.net(inputs)


class _RLvRLActorMLP(nn.Module):
    """Student inference path for RLvRLActorCritic checkpoints."""

    def __init__(self, state_dict, actor_obs_dim, history_obs_dim, latent_dim):
        super().__init__()
        self.adaptation_module = _PrefixedSequentialMLP(state_dict, "adaptation_module.")
        self.actor_body = _PrefixedSequentialMLP(state_dict, "actor_body.")
        self.input_dim = int(actor_obs_dim)
        self.history_input_dim = int(history_obs_dim)
        self.latent_dim = int(latent_dim)

        if self.adaptation_module.input_dim != self.history_input_dim:
            raise ValueError(
                "RLvRL adaptation input dim mismatch: "
                f"checkpoint={self.adaptation_module.input_dim} expected={self.history_input_dim}"
            )
        if self.adaptation_module.output_dim != self.latent_dim:
            raise ValueError(
                "RLvRL latent dim mismatch: "
                f"checkpoint={self.adaptation_module.output_dim} expected={self.latent_dim}"
            )
        expected_actor_body_input = self.input_dim + self.latent_dim
        if self.actor_body.input_dim != expected_actor_body_input:
            raise ValueError(
                "RLvRL actor body input dim mismatch: "
                f"checkpoint={self.actor_body.input_dim} expected={expected_actor_body_input}"
            )

    def forward(self, obs, history_obs):
        latent = self.adaptation_module(history_obs)
        return self.actor_body(torch.cat((obs, latent), dim=-1))


class _AdaptationMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


def _load_adaptation_network(path, device):
    checkpoint = torch.load(path, map_location=device)
    network = _AdaptationMLP(checkpoint["input_dim"], checkpoint["output_dim"]).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    network.requires_grad_(False)
    return network


class _ConcurrentExplicitEstimator(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state_dict = checkpoint["model_state_dict"]
        self.output_dim = int(checkpoint.get("output_dim", 12 if "base_height_head.weight" in state_dict else 11))
        if self.output_dim not in (11, 12):
            raise ValueError("Concurrent explicit estimator output_dim must be 11 or 12.")
        layer_ids = sorted(
            {
                int(key.removeprefix("trunk.").split(".")[0])
                for key in state_dict
                if key.startswith("trunk.") and key.endswith(".weight")
            }
        )
        if not layer_ids:
            raise ValueError("Cannot find concurrent explicit estimator trunk weights")

        modules = []
        for layer_id in layer_ids:
            weight = state_dict[f"trunk.{layer_id}.weight"]
            modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
            modules.append(nn.ELU())
        self.trunk = nn.Sequential(*modules)

        last_dim = int(state_dict[f"trunk.{layer_ids[-1]}.weight"].shape[0])
        self.velocity_head = nn.Linear(last_dim, 3)
        self.contact_head = nn.Linear(last_dim, 4)
        self.foot_height_head = nn.Linear(last_dim, 4)
        self.base_height_head = nn.Linear(last_dim, 1) if self.output_dim == 12 else None
        self.load_state_dict(state_dict)
        self.input_dim = int(checkpoint.get("input_dim", state_dict[f"trunk.{layer_ids[0]}.weight"].shape[1]))

    def forward(self, inputs):
        features = self.trunk(inputs)
        outputs = [
            self.velocity_head(features),
            self.contact_head(features),
            self.foot_height_head(features),
        ]
        if self.base_height_head is not None:
            outputs.append(self.base_height_head(features))
        return torch.cat(outputs, dim=-1)


def _load_concurrent_explicit_estimator(path, device):
    checkpoint = torch.load(path, map_location=device)
    network = _ConcurrentExplicitEstimator(checkpoint).to(device)
    network.eval()
    network.requires_grad_(False)
    return network


class Go2PosturePolicyWrapper:
    """MuJoCo-side wrapper for a go2_posture IsaacLab policy.

    This reconstructs the Go2-Posture-Direct-v0 actor observation from MuJoCo
    state and converts policy residual actions into joint position targets:
    processed_action = guide_action + action_scale * actor_action.
    """

    def __init__(self, env, run_dir=None, checkpoint=None, device=None, use_exported_adaptation=None):
        self.env = env
        self.run_dir = Path(run_dir or os.environ.get("GO2_POSTURE_RUN_DIR", DEFAULT_RUN_DIR)).expanduser()
        self.checkpoint_path = Path(
            checkpoint or os.environ.get("GO2_POSTURE_CHECKPOINT", self.run_dir / "model_9999.pt")
        ).expanduser()
        self.device = torch.device(device or os.environ.get("GO2_POSTURE_DEVICE", "cpu"))

        with open(self.run_dir / "params" / "env.yaml", "r") as file:
            self.cfg = yaml.unsafe_load(file)

        self.action_scale = float(self.cfg["action_scale"])
        self.history_length = int(self.cfg["obs_history_length"])
        self.base_obs_dim = int(self.cfg["base_observation_dim"])
        self.adaptation_obs_dim = int(self.cfg["adaptation_observation_dim"])
        self.use_rma = bool(self.cfg.get("use_rma", False))
        self.use_concurrent_state_estimator = bool(self.cfg.get("use_concurrent_state_estimator", False))
        self.concurrent_state_estimator_mode = str(self.cfg.get("concurrent_state_estimator_mode", "velocity")).lower()
        self.concurrent_policy_obs_mode = str(self.cfg.get("concurrent_policy_obs_mode", "current")).lower()
        if self.concurrent_policy_obs_mode not in {"current", "history"}:
            raise ValueError("concurrent_policy_obs_mode must be 'current' or 'history'")
        self.use_rlvrl_teacher_student = bool(self.cfg.get("use_rlvrl_teacher_student", False))
        self.RL_FREQ = 1.0 / (float(self.cfg["sim"]["dt"]) * float(self.cfg["decimation"]))
        self.joint_order = os.environ.get("GO2_POSTURE_JOINT_ORDER", "joint_grouped").strip().lower()
        if self.joint_order not in {"leg_grouped", "joint_grouped"}:
            raise ValueError("GO2_POSTURE_JOINT_ORDER must be 'leg_grouped' or 'joint_grouped'")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]
        if self.use_rlvrl_teacher_student:
            self.actor = _RLvRLActorMLP(
                state_dict,
                actor_obs_dim=int(self.cfg["rlvrl_actor_observation_dim"]),
                history_obs_dim=int(self.cfg["rlvrl_history_observation_dim"]),
                latent_dim=int(self.cfg["rlvrl_latent_dim"]),
            ).to(self.device)
        else:
            self.actor = _ActorMLP(state_dict).to(self.device)
        self.actor.eval()
        self.actor.requires_grad_(False)

        actuator_cfg = self.cfg["robot"]["actuators"]["base_legs"]
        self.Kp_walking = float(actuator_cfg.get("stiffness", 25.0))
        self.Kd_walking = float(actuator_cfg.get("damping", 0.5))
        self.Kp_stand_up_and_down = float(os.environ.get("GO2_POSTURE_STAND_KP", "25.0"))
        self.Kd_stand_up_and_down = float(os.environ.get("GO2_POSTURE_STAND_KD", "2.0"))
        self.effort_limit = float(actuator_cfg.get("effort_limit", np.inf))

        self.default_joint_pos = np.array(
            [
                0.1, 0.8, -1.5,
                -0.1, 0.8, -1.5,
                0.1, 1.0, -1.5,
                -0.1, 1.0, -1.5,
            ],
            dtype=np.float32,
        )
        self.default_joint_pos_policy = self._to_policy_order(self.default_joint_pos)
        self.initial_base_height = float(self.cfg["robot"]["init_state"]["pos"][2])

        self.obs_history = np.zeros((self.history_length, self.base_obs_dim), dtype=np.float32)
        self.adaptation_history = np.zeros((self.history_length, self.adaptation_obs_dim), dtype=np.float32)
        self.concurrent_estimator_single_obs_dim = int(self.cfg.get("concurrent_estimator_single_observation_dim", 45))
        self.concurrent_estimator_history_length = int(self.cfg.get("concurrent_estimator_history_length", 1))
        self.concurrent_explicit_state_dim = int(self.cfg.get("concurrent_explicit_estimator_output_dim", 11))
        self.concurrent_estimator_history = np.zeros(
            (self.concurrent_estimator_history_length, self.concurrent_estimator_single_obs_dim),
            dtype=np.float32,
        )
        self.cse_previous_joint_target_policy = self.default_joint_pos_policy.copy()
        self.cse_llast_joint_target_policy = self.default_joint_pos_policy.copy()
        self.cse_joint_pos_err_history = np.zeros((3, 12), dtype=np.float32)
        self.cse_joint_vel_history = np.zeros((3, 12), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.desired_joint_pos = _legs_attr_from_flat(self.default_joint_pos)
        if self.concurrent_state_estimator_mode == "explicit":
            self.policy_obs_dim = self.base_obs_dim
        else:
            self.policy_obs_dim = self.base_obs_dim * self.history_length
        if self.use_rma:
            self.policy_obs_dim += int(self.cfg.get("rma_output_dim", 11))
        self.rlvrl_actor_obs_mode = None

        self.prev_vx = 0.0
        self.a_long_filtered = 0.0
        self.guide_roll = 0.0
        self.guide_pitch = 0.0
        self.guide_height = float(self.cfg["guide_h_nom"])
        self.guide_action = self.default_joint_pos.copy()
        self.guide_action_policy = self.default_joint_pos_policy.copy()

        if use_exported_adaptation is None:
            self.use_exported_adaptation = os.environ.get("GO2_POSTURE_USE_EXPORTED_ADAPTATION", "0") == "1"
        else:
            self.use_exported_adaptation = bool(use_exported_adaptation)
        self.concurrent_state_estimator = None
        self.concurrent_explicit_estimator = None
        self.rma_network = None
        if self.use_exported_adaptation:
            exported = self.run_dir / "exported"
            if self.use_concurrent_state_estimator:
                if self.concurrent_state_estimator_mode == "explicit":
                    self.concurrent_explicit_estimator = _load_concurrent_explicit_estimator(
                        exported / "concurrent_explicit_estimator.pth", self.device
                    )
                else:
                    self.concurrent_state_estimator = _load_adaptation_network(
                        exported / "concurrent_state_estimator.pth", self.device
                    )
            if self.use_rma:
                self.rma_network = _load_adaptation_network(exported / "rma.pth", self.device)

        if self.use_rlvrl_teacher_student:
            expected_history_dim = self.adaptation_obs_dim * self.history_length
            if self.actor.history_input_dim != expected_history_dim:
                raise ValueError(
                    "go2_posture RLvRL history input dim mismatch: "
                    f"actor={self.actor.history_input_dim} expected={expected_history_dim}"
                )
            if self.actor.input_dim == self.base_obs_dim:
                self.rlvrl_actor_obs_mode = "current"
            elif self.actor.input_dim == self.policy_obs_dim:
                self.rlvrl_actor_obs_mode = "history"
            else:
                raise ValueError(
                    "go2_posture RLvRL actor obs dim is unsupported by this wrapper: "
                    f"actor={self.actor.input_dim} current={self.base_obs_dim} history={self.policy_obs_dim}"
                )
        elif self.actor.input_dim != self.policy_obs_dim:
            raise ValueError(
                f"go2_posture actor input dim mismatch: actor={self.actor.input_dim} expected={self.policy_obs_dim}"
            )

        print(
            "[go2_posture] loaded "
            f"{self.checkpoint_path} obs_dim={self.actor.input_dim} "
            f"RL_FREQ={self.RL_FREQ:.1f}Hz Kp={self.Kp_walking:.2f} Kd={self.Kd_walking:.2f} "
            f"joint_order={self.joint_order} rlvrl={self.use_rlvrl_teacher_student}"
            f" rlvrl_actor_obs={self.rlvrl_actor_obs_mode}"
            f" concurrent_mode={self.concurrent_state_estimator_mode}"
        )
        if not self.use_exported_adaptation:
            print("[go2_posture] MuJoCo oracle adaptation is enabled for sim-to-sim.")

    def compute_control(
        self,
        base_pos,
        base_quat_wxyz,
        base_lin_vel,
        base_ang_vel,
        joints_pos,
        joints_vel,
        ref_base_lin_vel,
        ref_base_ang_vel,
        feet_contacts=None,
    ):
        base_lin_vel = np.asarray(base_lin_vel, dtype=np.float32).reshape(3)
        base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32).reshape(3)
        command = np.array([ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_ang_vel[2]], dtype=np.float32)
        joint_pos_leg_grouped = _flat_legs(joints_pos)
        joint_vel_leg_grouped = _flat_legs(joints_vel)
        joint_pos = self._to_policy_order(joint_pos_leg_grouped)
        joint_vel = self._to_policy_order(joint_vel_leg_grouped)

        prev_guide_obs = np.array([self.guide_roll, self.guide_pitch, self.guide_height], dtype=np.float32)
        adaptation_obs = np.concatenate(
            [
                base_ang_vel,
                self._get_projected_gravity(base_quat_wxyz),
                command,
                joint_pos - self.default_joint_pos_policy,
                joint_vel,
                self.last_action,
                prev_guide_obs,
            ]
        ).astype(np.float32)
        adaptation_obs_flat = self._append_adaptation_history(adaptation_obs)

        if self.concurrent_state_estimator_mode == "explicit":
            estimator_obs = self._build_concurrent_estimator_observation(
                base_quat_wxyz=base_quat_wxyz,
                base_ang_vel=base_ang_vel,
                command=command,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
            )
            if estimator_obs.shape[0] != self.concurrent_estimator_single_obs_dim:
                raise ValueError(
                    "concurrent explicit estimator obs dim mismatch: "
                    f"got={estimator_obs.shape[0]} expected={self.concurrent_estimator_single_obs_dim}"
                )
            estimator_obs_flat = self._append_concurrent_estimator_history(estimator_obs)
            if self.concurrent_explicit_estimator is not None:
                with torch.no_grad():
                    raw_explicit_state = (
                        self.concurrent_explicit_estimator(
                            torch.tensor(estimator_obs_flat, dtype=torch.float32, device=self.device).view(1, -1)
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                explicit_state_obs = self._process_concurrent_explicit_output(raw_explicit_state)
            else:
                explicit_state_obs = self._oracle_concurrent_explicit_state(base_lin_vel, feet_contacts, base_pos)
            root_lin_vel_obs = explicit_state_obs[:3]
        elif self.concurrent_state_estimator is not None:
            with torch.no_grad():
                root_lin_vel_obs = (
                    self.concurrent_state_estimator(
                        torch.tensor(adaptation_obs_flat, dtype=torch.float32, device=self.device).view(1, -1)
                    )
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            explicit_state_obs = None
            estimator_obs = None
            estimator_obs_flat = None
        else:
            root_lin_vel_obs = base_lin_vel
            explicit_state_obs = None
            estimator_obs = None
            estimator_obs_flat = None

        self._update_guide(command, root_lin_vel_obs[0], velocity_b=root_lin_vel_obs, omega_b=base_ang_vel)
        guide_obs = np.array([self.guide_roll, self.guide_pitch, self.guide_height], dtype=np.float32)

        if self.concurrent_state_estimator_mode == "explicit":
            actor_base_obs = estimator_obs_flat if self.concurrent_policy_obs_mode == "history" else estimator_obs
            obs_step = np.concatenate([actor_base_obs, explicit_state_obs, guide_obs]).astype(np.float32)
        else:
            obs_step = np.concatenate(
                [
                    root_lin_vel_obs.astype(np.float32),
                    base_ang_vel,
                    self._get_projected_gravity(base_quat_wxyz),
                    command,
                    joint_pos - self.default_joint_pos_policy,
                    joint_vel,
                    self.last_action,
                    guide_obs,
                ]
            ).astype(np.float32)
        if obs_step.shape[0] != self.base_obs_dim:
            raise ValueError(f"go2_posture obs_step dim mismatch: got={obs_step.shape[0]} expected={self.base_obs_dim}")
        obs_history_flat = self._append_obs_history(obs_step)
        obs = obs_step.copy() if self.concurrent_state_estimator_mode == "explicit" else obs_history_flat

        if self.use_rma:
            if self.rma_network is not None:
                with torch.no_grad():
                    rma_obs = (
                        self.rma_network(
                            torch.tensor(adaptation_obs_flat, dtype=torch.float32, device=self.device).view(1, -1)
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            else:
                rma_obs = self._privileged_observation(base_lin_vel, base_pos, base_quat_wxyz, feet_contacts)
            obs = np.concatenate([obs, rma_obs.astype(np.float32)], axis=0)

        with torch.no_grad():
            if self.use_rlvrl_teacher_student:
                actor_obs = obs_step if self.rlvrl_actor_obs_mode == "current" else obs
                obs_t = torch.tensor(actor_obs, dtype=torch.float32, device=self.device).view(1, -1)
                history_t = torch.tensor(adaptation_obs_flat, dtype=torch.float32, device=self.device).view(1, -1)
                action_t = self.actor(obs_t, history_t)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
                action_t = self.actor(obs_t)
            action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        self.last_action = action.copy()
        joint_target_policy = self.guide_action_policy + self.action_scale * action
        self.cse_llast_joint_target_policy = self.cse_previous_joint_target_policy.copy()
        self.cse_previous_joint_target_policy = joint_target_policy.copy()
        joint_target = self._to_leg_grouped(joint_target_policy)
        self.desired_joint_pos = _legs_attr_from_flat(joint_target)
        return self.desired_joint_pos

    def _build_concurrent_estimator_observation(self, base_quat_wxyz, base_ang_vel, command, joint_pos, joint_vel):
        projected_gravity = self._get_projected_gravity(base_quat_wxyz)
        joint_pos_err = joint_pos - self.default_joint_pos_policy
        if self.concurrent_estimator_single_obs_dim == 45:
            return np.concatenate(
                [
                    base_ang_vel,
                    projected_gravity,
                    command,
                    joint_pos_err,
                    joint_vel,
                    self.last_action,
                ]
            ).astype(np.float32)

        previous_joint_target_err = self.cse_previous_joint_target_policy - self.default_joint_pos_policy
        llast_joint_target_err = self.cse_llast_joint_target_policy - self.default_joint_pos_policy
        foot_pos_b = self._get_foot_positions_body_frame_policy_order()
        obs = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                command,
                joint_pos_err,
                joint_vel,
                previous_joint_target_err,
                llast_joint_target_err,
                self.cse_joint_pos_err_history.reshape(-1),
                self.cse_joint_vel_history.reshape(-1),
                foot_pos_b,
            ]
        ).astype(np.float32)
        self.cse_joint_pos_err_history[:-1] = self.cse_joint_pos_err_history[1:]
        self.cse_joint_pos_err_history[-1] = joint_pos_err
        self.cse_joint_vel_history[:-1] = self.cse_joint_vel_history[1:]
        self.cse_joint_vel_history[-1] = joint_vel
        return obs

    def _get_foot_positions_body_frame_policy_order(self):
        try:
            return _flat_legs(self.env.feet_pos(frame="base"))
        except Exception:
            return np.zeros(12, dtype=np.float32)

    def _append_obs_history(self, obs_step):
        self.obs_history[:-1] = self.obs_history[1:]
        self.obs_history[-1] = obs_step
        return self.obs_history.reshape(-1).astype(np.float32)

    def _append_adaptation_history(self, obs_step):
        self.adaptation_history[:-1] = self.adaptation_history[1:]
        self.adaptation_history[-1] = obs_step
        return self.adaptation_history.reshape(-1).astype(np.float32)

    def _append_concurrent_estimator_history(self, obs_step):
        self.concurrent_estimator_history[:-1] = self.concurrent_estimator_history[1:]
        self.concurrent_estimator_history[-1] = obs_step
        return self.concurrent_estimator_history.reshape(-1).astype(np.float32)

    def _process_concurrent_explicit_output(self, raw_state):
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(self.concurrent_explicit_state_dim)
        velocity = raw_state[:3].copy()
        vx_min, vx_max = self.cfg.get("concurrent_velocity_clamp_x", (-4.0, 4.0))
        vy_min, vy_max = self.cfg.get("concurrent_velocity_clamp_y", (-2.0, 2.0))
        vz_min, vz_max = self.cfg.get("concurrent_velocity_clamp_z", (-1.0, 1.0))
        velocity[0] = np.clip(velocity[0], float(vx_min), float(vx_max))
        velocity[1] = np.clip(velocity[1], float(vy_min), float(vy_max))
        velocity[2] = np.clip(velocity[2], float(vz_min), float(vz_max))
        contacts = 1.0 / (1.0 + np.exp(-np.clip(raw_state[3:7], -60.0, 60.0)))
        foot_height = raw_state[7:11].copy()
        fh_min, fh_max = self.cfg.get("concurrent_foot_height_clamp", (-0.2, 0.5))
        foot_height = np.clip(foot_height, float(fh_min), float(fh_max))
        state = [velocity, contacts, foot_height]
        if raw_state.shape[0] >= 12:
            bh_min, bh_max = self.cfg.get("concurrent_base_height_clamp", (0.12, 0.7))
            base_height = np.array([np.clip(raw_state[11], float(bh_min), float(bh_max))], dtype=np.float32)
            state.append(base_height)
        return np.concatenate(state).astype(np.float32)

    def _oracle_concurrent_explicit_state(self, base_lin_vel, feet_contacts, base_pos=None):
        velocity = np.asarray(base_lin_vel, dtype=np.float32).reshape(3)
        if feet_contacts is None:
            contacts = np.zeros(4, dtype=np.float32)
        else:
            contacts = np.asarray(feet_contacts, dtype=np.float32).reshape(4)
        foot_height = np.zeros(4, dtype=np.float32)
        state = [velocity, contacts, foot_height]
        if self.concurrent_explicit_state_dim >= 12:
            if base_pos is None:
                base_height_value = float(self.cfg.get("concurrent_explicit_base_height_init", self.guide_height))
            else:
                base_height_value = float(np.asarray(base_pos, dtype=np.float32).reshape(3)[2])
            bh_min, bh_max = self.cfg.get("concurrent_base_height_clamp", (0.12, 0.7))
            state.append(np.array([np.clip(base_height_value, float(bh_min), float(bh_max))], dtype=np.float32))
        return np.concatenate(state).astype(np.float32)

    def _update_guide(self, command, current_vx, velocity_b=None, omega_b=None):
        ctrl_dt = 1.0 / self.RL_FREQ
        raw_a_long = (float(current_vx) - self.prev_vx) / ctrl_dt
        alpha = float(self.cfg["guide_a_long_ema_alpha"])
        self.a_long_filtered = alpha * raw_a_long + (1.0 - alpha) * self.a_long_filtered
        self.prev_vx = float(current_vx)

        g = float(self.cfg["guide_gravity"])
        velocity_xy = np.asarray(command[:2] if velocity_b is None else velocity_b[:2], dtype=np.float32)
        yaw_rate = float(command[2] if omega_b is None else omega_b[2])
        vel_mag = float(np.linalg.norm(velocity_xy))
        abs_wz = abs(yaw_rate)
        a_lat = vel_mag * abs_wz

        roll_raw = float(self.cfg["guide_k_roll"]) * np.arctan(a_lat / g)
        turn_sign = float(command[2]) if abs(float(command[2])) > 1e-6 else yaw_rate
        roll_signed = -roll_raw if turn_sign > 0.0 else roll_raw
        if abs_wz < 1e-6 or vel_mag < 1e-6:
            roll_signed = 0.0

        pitch_raw = float(self.cfg["guide_k_pitch"]) * np.arctan(self.a_long_filtered / g)
        pitch = -pitch_raw
        height = (
            float(self.cfg["guide_h_nom"])
            - float(self.cfg["guide_kh_lat"]) * a_lat
            - float(self.cfg["guide_kh_long"]) * abs(self.a_long_filtered)
            - float(self.cfg["guide_kh_speed"]) * vel_mag**2
        )
        if bool(self.cfg.get("guide_clamp_enabled", True)):
            roll_signed = float(np.clip(roll_signed, -float(self.cfg["guide_roll_max"]), float(self.cfg["guide_roll_max"])))
            pitch = float(np.clip(pitch, -float(self.cfg["guide_pitch_max"]), float(self.cfg["guide_pitch_max"])))
            height = float(np.clip(height, float(self.cfg["guide_h_min"]), float(self.cfg["guide_h_nom"])))

        self.guide_roll = roll_signed
        self.guide_pitch = pitch
        self.guide_height = height
        self.guide_action = self._posture_to_joint_targets(roll_signed, pitch, height)
        self.guide_action_policy = self._to_policy_order(self.guide_action)

    def _posture_to_joint_targets(self, roll, pitch, height):
        delta = np.zeros(12, dtype=np.float32)
        delta[[0, 3, 6, 9]] = float(self.cfg["k_roll_hip"]) * roll
        delta[[1, 4]] = float(self.cfg["k_pitch_thigh"]) * pitch
        delta[[7, 10]] = -float(self.cfg["k_pitch_thigh"]) * pitch

        delta_h = height - float(self.cfg["guide_h_nom"])
        delta[[1, 4, 7, 10]] += -float(self.cfg["k_h_thigh"]) * delta_h
        delta[[2, 5, 8, 11]] = float(self.cfg["k_h_calf"]) * delta_h
        return self.default_joint_pos + delta

    def _to_policy_order(self, q_leg_grouped):
        if self.joint_order == "joint_grouped":
            return _leg_grouped_to_joint_grouped(q_leg_grouped)
        return np.asarray(q_leg_grouped, dtype=np.float32).reshape(12)

    def _to_leg_grouped(self, q_policy_order):
        if self.joint_order == "joint_grouped":
            return _joint_grouped_to_leg_grouped(q_policy_order)
        return np.asarray(q_policy_order, dtype=np.float32).reshape(12)

    def _privileged_observation(self, base_lin_vel, base_pos, base_quat_wxyz, feet_contacts):
        roll, pitch = self._roll_pitch_from_quat(base_quat_wxyz)
        base_height = float(np.asarray(base_pos).reshape(3)[2])
        guide_errors = np.array(
            [
                roll - self.guide_roll,
                pitch - self.guide_pitch,
                base_height - self.guide_height,
            ],
            dtype=np.float32,
        )
        if feet_contacts is None:
            contacts = np.zeros(4, dtype=np.float32)
        else:
            contacts = np.asarray(feet_contacts, dtype=np.float32).reshape(4)
        return np.concatenate([base_lin_vel, [base_height], guide_errors, contacts]).astype(np.float32)

    @staticmethod
    def _roll_pitch_from_quat(q_wxyz):
        q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
        w, x, y, z = q / max(float(np.linalg.norm(q)), 1.0e-8)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
        return float(roll), float(pitch)

    @staticmethod
    def _get_projected_gravity(quat_wxyz):
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        w, x, y, z = q / max(float(np.linalg.norm(q)), 1.0e-8)
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        q_vec = np.array([x, y, z], dtype=np.float32)
        projected = (
            gravity * (2.0 * w * w - 1.0)
            - 2.0 * w * np.cross(q_vec, gravity)
            + 2.0 * q_vec * np.dot(q_vec, gravity)
        )
        return projected.astype(np.float32)
