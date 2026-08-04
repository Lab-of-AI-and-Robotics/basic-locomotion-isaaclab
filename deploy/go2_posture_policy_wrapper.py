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


def _wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _infer_guide_obs_dim(cfg, base_obs_dim, adaptation_obs_dim):
    if not bool(cfg.get("posture_guide_arch_enabled", True)):
        return 0
    explicit_dim = cfg.get("posture_guide_observation_dim")
    if explicit_dim in (0, 1, 3):
        return int(explicit_dim)

    # Deployable obs: ang_vel(3) + projected_gravity(3) + command(3)
    # + joint_pos_err(12) + joint_vel(12) + previous_action(12) = 45, plus guide.
    guide_dim = int(adaptation_obs_dim) - 45
    if guide_dim in (0, 1, 3):
        return guide_dim

    # Velocity/current obs: root_lin_vel(3) + deployable obs without guide = 48, plus guide.
    guide_dim = int(base_obs_dim) - 48
    if guide_dim in (0, 1, 3):
        return guide_dim

    return 3


def _go2_forward_feet_in_base(
    joint_pos,
    hip_x_front=0.1934,
    hip_x_rear=-0.1934,
    hip_y=0.0465,
    thigh_offset_y=0.0955,
    thigh_length=0.213,
    calf_length=0.213,
):
    q = np.asarray(joint_pos, dtype=np.float32).reshape(-1, 4, 3)
    q_hip = q[..., 0]
    q_thigh = q[..., 1]
    q_calf = q[..., 2]

    leg_side = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32).reshape(1, 4)
    hip_origin = np.array(
        [
            [hip_x_front, hip_y, 0.0],
            [hip_x_front, -hip_y, 0.0],
            [hip_x_rear, hip_y, 0.0],
            [hip_x_rear, -hip_y, 0.0],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 3)

    sagittal_x = -thigh_length * np.sin(q_thigh) - calf_length * np.sin(q_thigh + q_calf)
    sagittal_z = -thigh_length * np.cos(q_thigh) - calf_length * np.cos(q_thigh + q_calf)
    lateral_y = leg_side * thigh_offset_y

    cos_hip = np.cos(q_hip)
    sin_hip = np.sin(q_hip)
    rel_x = sagittal_x
    rel_y = cos_hip * lateral_y - sin_hip * sagittal_z
    rel_z = sin_hip * lateral_y + cos_hip * sagittal_z

    return hip_origin + np.stack((rel_x, rel_y, rel_z), axis=-1)


def _posture_to_joint_targets_ik(
    roll,
    pitch,
    height,
    default_joint_pos,
    h_nom,
    lateral_shift=None,
    hip_x_front=0.1934,
    hip_x_rear=-0.1934,
    hip_y=0.0465,
    thigh_offset_y=0.0955,
    thigh_length=0.213,
    calf_length=0.213,
):
    default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(1, 12)
    roll = np.asarray([roll], dtype=np.float32).reshape(-1, 1)
    pitch = np.asarray([pitch], dtype=np.float32).reshape(-1, 1)
    height = np.asarray([height], dtype=np.float32).reshape(-1, 1)

    nominal_feet_b = _go2_forward_feet_in_base(
        default_joint_pos,
        hip_x_front=hip_x_front,
        hip_x_rear=hip_x_rear,
        hip_y=hip_y,
        thigh_offset_y=thigh_offset_y,
        thigh_length=thigh_length,
        calf_length=calf_length,
    )
    world_rel = nominal_feet_b.copy()
    if lateral_shift is not None:
        lateral_shift = np.asarray([lateral_shift], dtype=np.float32).reshape(-1, 1)
        world_rel[..., 1] -= lateral_shift
    world_rel[..., 2] += float(h_nom) - height

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    x_w = world_rel[..., 0]
    y_w = world_rel[..., 1]
    z_w = world_rel[..., 2]
    x_p = cp * x_w - sp * z_w
    y_p = y_w
    z_p = sp * x_w + cp * z_w
    target_feet_b = np.stack((x_p, cr * y_p + sr * z_p, -sr * y_p + cr * z_p), axis=-1)

    leg_side = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32).reshape(1, 4)
    hip_origin = np.array(
        [
            [hip_x_front, hip_y, 0.0],
            [hip_x_front, -hip_y, 0.0],
            [hip_x_rear, hip_y, 0.0],
            [hip_x_rear, -hip_y, 0.0],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 3)
    foot_h = target_feet_b - hip_origin

    x = foot_h[..., 0]
    y = foot_h[..., 1]
    z = foot_h[..., 2]
    lateral = leg_side * float(thigh_offset_y)
    yz_sq = np.square(y) + np.square(z)
    sagittal_z = -np.sqrt(np.clip(yz_sq - float(thigh_offset_y) ** 2, 1.0e-8, None))

    q_hip = _wrap_to_pi(np.arctan2(z, y) - np.arctan2(sagittal_z, lateral))
    reach_sq = np.clip(np.square(x) + np.square(sagittal_z), 1.0e-8, None)
    cos_knee = np.clip(
        (reach_sq - float(thigh_length) ** 2 - float(calf_length) ** 2)
        / (2.0 * float(thigh_length) * float(calf_length)),
        -0.999999,
        0.999999,
    )
    q_calf = -np.arccos(cos_knee)
    q_thigh = np.arctan2(-x, -sagittal_z) - np.arctan2(
        float(calf_length) * np.sin(q_calf),
        float(thigh_length) + float(calf_length) * np.cos(q_calf),
    )

    q_hip = np.clip(q_hip, -1.0472, 1.0472)
    thigh_lower = np.array([-1.5708, -1.5708, -0.5236, -0.5236], dtype=np.float32).reshape(1, 4)
    thigh_upper = np.array([3.4907, 3.4907, 4.5379, 4.5379], dtype=np.float32).reshape(1, 4)
    q_thigh = np.minimum(np.maximum(q_thigh, thigh_lower), thigh_upper)
    q_calf = np.clip(q_calf, -2.7227, -0.83776)

    return np.stack((q_hip, q_thigh, q_calf), axis=-1).reshape(12).astype(np.float32)


def _resolve_activation(name):
    name = str(name or "elu").lower()
    if name == "elu":
        return nn.ELU
    if name == "selu":
        return nn.SELU
    if name == "relu":
        return nn.ReLU
    if name == "crelu":
        return nn.CELU
    if name == "lrelu":
        return nn.LeakyReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "identity":
        return nn.Identity
    raise ValueError(f"Invalid activation function {name!r}")


class _ActorMLP(nn.Module):
    def __init__(self, state_dict, activation="elu"):
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
                modules.append(_resolve_activation(activation)())
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


def _default_joint_mirror_perm_and_sign(joint_order):
    if joint_order == "joint_grouped":
        return (
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    return (
        [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    )


class _Go2EquivariantActorMLP(nn.Module):
    """Inference-only port of Go2EquivariantActorInvariantCritic.act_inference()."""

    def __init__(self, state_dict, cfg, joint_order, policy_cfg=None):
        super().__init__()
        policy_cfg = policy_cfg or {}
        self.activation = str(policy_cfg.get("activation", "elu"))
        self.actor = _ActorMLP(state_dict, activation=self.activation)
        self.input_dim = self.actor.input_dim

        joint_perm = policy_cfg.get("joint_mirror_perm")
        joint_sign = policy_cfg.get("joint_mirror_sign")
        if joint_perm is None or joint_sign is None:
            joint_perm, joint_sign = _default_joint_mirror_perm_and_sign(joint_order)
        feet_name_perm = policy_cfg.get("feet_name_mirror_perm") or [1, 0, 3, 2]
        feet_body_perm = policy_cfg.get("feet_body_mirror_perm") or [1, 0, 3, 2]

        self.register_buffer("joint_mirror_perm", torch.tensor(joint_perm, dtype=torch.long), persistent=False)
        self.register_buffer("joint_mirror_sign", torch.tensor(joint_sign, dtype=torch.float32), persistent=False)
        self.register_buffer("feet_name_mirror_perm", torch.tensor(feet_name_perm, dtype=torch.long), persistent=False)
        self.register_buffer("feet_body_mirror_perm", torch.tensor(feet_body_perm, dtype=torch.long), persistent=False)

        self.base_observation_dim = int(cfg.get("base_observation_dim", 51))
        self.obs_history_length = int(cfg.get("obs_history_length", 1))
        self.critic_privileged_observation_dim = int(cfg.get("critic_privileged_observation_dim", 11))
        self.concurrent_estimator_single_observation_dim = int(
            cfg.get("concurrent_estimator_single_observation_dim", 141)
        )
        self.concurrent_explicit_estimator_output_dim = int(
            cfg.get("concurrent_explicit_estimator_output_dim", 12)
        )
        self.posture_guide_arch_enabled = bool(cfg.get("posture_guide_arch_enabled", True))
        self.guide_obs_dim = _infer_guide_obs_dim(
            cfg,
            self.base_observation_dim,
            int(cfg.get("adaptation_observation_dim", self.base_observation_dim)),
        )

    def forward(self, obs):
        mean = self.actor(obs)
        mirrored_obs = self._mirror_obs(obs, "policy")
        mirrored_mean = self._mirror_joint_block(self.actor(mirrored_obs))
        return 0.5 * (mean + mirrored_mean)

    def _mirror_joint_block(self, data):
        return data.index_select(-1, self.joint_mirror_perm) * self.joint_mirror_sign.to(data.dtype)

    def _mirror_leg_scalar_block(self, data, body_order):
        if data.shape[-1] != 4:
            return data
        perm = self.feet_body_mirror_perm if body_order else self.feet_name_mirror_perm
        return data.index_select(-1, perm)

    def _mirror_foot_position_block(self, data):
        if data.shape[-1] != 12:
            return data
        feet = data.reshape(*data.shape[:-1], 4, 3)
        feet = feet.index_select(-2, self.feet_body_mirror_perm)
        feet = feet * feet.new_tensor([1.0, -1.0, 1.0])
        return feet.reshape_as(data)

    @staticmethod
    def _mirror_vec3(data, kind):
        if kind == "polar":
            sign = data.new_tensor([1.0, -1.0, 1.0])
        elif kind == "axial":
            sign = data.new_tensor([-1.0, 1.0, -1.0])
        elif kind == "command":
            sign = data.new_tensor([1.0, -1.0, -1.0])
        else:
            raise ValueError(f"Unknown mirror vec3 kind: {kind}")
        return data * sign

    def _mirror_guide(data):
        if data.shape[-1] == 1:
            return -data
        if data.shape[-1] == 3:
            return data * data.new_tensor([-1.0, 1.0, 1.0])
        raise ValueError(f"Unsupported guide width for PPOeqic: {data.shape[-1]}")

    def _mirror_explicit_state(self, data):
        out = data.clone()
        out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "polar")
        out[..., 3:7] = self._mirror_leg_scalar_block(out[..., 3:7], body_order=False)
        out[..., 7:11] = self._mirror_leg_scalar_block(out[..., 7:11], body_order=True)
        return out

    def _mirror_estimator_obs(self, data):
        out = data.clone()
        out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "axial")
        out[..., 3:6] = self._mirror_vec3(out[..., 3:6], "polar")
        out[..., 6:9] = self._mirror_vec3(out[..., 6:9], "command")
        out[..., 9:21] = self._mirror_joint_block(out[..., 9:21])
        out[..., 21:33] = self._mirror_joint_block(out[..., 21:33])
        out[..., 33:45] = self._mirror_joint_block(out[..., 33:45])
        out[..., 45:57] = self._mirror_joint_block(out[..., 45:57])
        offset = 57
        for _ in range(3):
            out[..., offset : offset + 12] = self._mirror_joint_block(out[..., offset : offset + 12])
            offset += 12
        for _ in range(3):
            out[..., offset : offset + 12] = self._mirror_joint_block(out[..., offset : offset + 12])
            offset += 12
        out[..., 129:141] = self._mirror_foot_position_block(out[..., 129:141])
        return out

    def _mirror_base_obs(self, data):
        out = data.clone()
        if data.shape[-1] in (49, 51):
            out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "polar")
            out[..., 3:6] = self._mirror_vec3(out[..., 3:6], "axial")
            out[..., 6:9] = self._mirror_vec3(out[..., 6:9], "polar")
            out[..., 9:12] = self._mirror_vec3(out[..., 9:12], "command")
            out[..., 12:24] = self._mirror_joint_block(out[..., 12:24])
            out[..., 24:36] = self._mirror_joint_block(out[..., 24:36])
            out[..., 36:48] = self._mirror_joint_block(out[..., 36:48])
            out[..., 48:] = self._mirror_guide(out[..., 48:])
        elif data.shape[-1] in (45, 46, 48):
            out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "axial")
            out[..., 3:6] = self._mirror_vec3(out[..., 3:6], "polar")
            out[..., 6:9] = self._mirror_vec3(out[..., 6:9], "command")
            out[..., 9:21] = self._mirror_joint_block(out[..., 9:21])
            out[..., 21:33] = self._mirror_joint_block(out[..., 21:33])
            out[..., 33:45] = self._mirror_joint_block(out[..., 33:45])
            if data.shape[-1] > 45:
                out[..., 45:] = self._mirror_guide(out[..., 45:])
        else:
            raise ValueError(f"Unsupported Go2 posture base observation width for PPOeqic: {data.shape[-1]}")
        return out

    def _mirror_privileged_obs(self, data):
        out = data.clone()
        if data.shape[-1] == 11:
            out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "polar")
            out[..., 4:7] = self._mirror_guide(out[..., 4:7])
            out[..., 7:11] = self._mirror_leg_scalar_block(out[..., 7:11], body_order=False)
        elif data.shape[-1] == 8:
            out[..., 0:3] = self._mirror_vec3(out[..., 0:3], "polar")
            out[..., 4:8] = self._mirror_leg_scalar_block(out[..., 4:8], body_order=False)
        else:
            raise ValueError(f"Unsupported Go2 posture privileged observation width for PPOeqic: {data.shape[-1]}")
        return out

    def _mirror_obs(self, obs, obs_type):
        width = obs.shape[-1]
        guide_dim = self.guide_obs_dim
        estimator_dim = self.concurrent_estimator_single_observation_dim
        explicit_dim = self.concurrent_explicit_estimator_output_dim

        if width == estimator_dim + explicit_dim + guide_dim:
            out = obs.clone()
            out[..., :estimator_dim] = self._mirror_estimator_obs(out[..., :estimator_dim])
            out[..., estimator_dim : estimator_dim + explicit_dim] = self._mirror_explicit_state(
                out[..., estimator_dim : estimator_dim + explicit_dim]
            )
            if guide_dim:
                out[..., -guide_dim:] = self._mirror_guide(out[..., -guide_dim:])
            return out

        if width in (46, 48, 49, 51):
            return self._mirror_base_obs(obs)

        if self.obs_history_length > 1 and width == self.base_observation_dim * self.obs_history_length:
            return self._mirror_base_obs(
                obs.reshape(*obs.shape[:-1], self.obs_history_length, self.base_observation_dim)
            ).reshape_as(obs)

        if obs_type == "critic":
            priv_dim = self.critic_privileged_observation_dim
            if width == priv_dim:
                return self._mirror_privileged_obs(obs)
            if width > priv_dim:
                out = obs.clone()
                out[..., :-priv_dim] = self._mirror_obs(out[..., :-priv_dim], "policy")
                out[..., -priv_dim:] = self._mirror_privileged_obs(out[..., -priv_dim:])
                return out

        raise ValueError(f"Unsupported Go2 posture {obs_type} observation width for PPOeqic: {width}")


class _PrefixedSequentialMLP(nn.Module):
    def __init__(self, state_dict, prefix, activation="elu"):
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
                modules.append(_resolve_activation(activation)())
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


class _RLvRLActorMLP(_Go2EquivariantActorMLP):
    """Student inference path for RLvRLActorCritic checkpoints."""

    def __init__(
        self,
        state_dict,
        actor_obs_dim,
        history_obs_dim,
        latent_dim,
        activation="elu",
        equivariant=False,
        cfg=None,
        joint_order="joint_grouped",
        policy_cfg=None,
    ):
        nn.Module.__init__(self)
        cfg = cfg or {}
        policy_cfg = policy_cfg or {}
        self.activation = str(activation)
        self.equivariant = bool(equivariant)
        self.adaptation_module = _PrefixedSequentialMLP(
            state_dict, "adaptation_module.", activation=self.activation
        )
        self.actor_body = _PrefixedSequentialMLP(state_dict, "actor_body.", activation=self.activation)
        self.input_dim = int(actor_obs_dim)
        self.history_input_dim = int(history_obs_dim)
        self.latent_dim = int(latent_dim)
        self.obs_history_length = int(cfg.get("obs_history_length", 1))
        self.base_observation_dim = int(cfg.get("base_observation_dim", self.input_dim))
        self.critic_privileged_observation_dim = int(cfg.get("critic_privileged_observation_dim", 11))
        self.concurrent_estimator_single_observation_dim = int(
            cfg.get("concurrent_estimator_single_observation_dim", 141)
        )
        self.concurrent_explicit_estimator_output_dim = int(
            cfg.get("concurrent_explicit_estimator_output_dim", 12)
        )
        self.posture_guide_arch_enabled = bool(cfg.get("posture_guide_arch_enabled", True))
        self.guide_obs_dim = _infer_guide_obs_dim(
            cfg,
            self.base_observation_dim,
            int(cfg.get("adaptation_observation_dim", self.base_observation_dim)),
        )

        if self.equivariant:
            joint_perm = policy_cfg.get("joint_mirror_perm")
            joint_sign = policy_cfg.get("joint_mirror_sign")
            if joint_perm is None or joint_sign is None:
                joint_perm, joint_sign = _default_joint_mirror_perm_and_sign(joint_order)
            feet_name_perm = policy_cfg.get("feet_name_mirror_perm") or [1, 0, 3, 2]
            feet_body_perm = policy_cfg.get("feet_body_mirror_perm") or [1, 0, 3, 2]

            self.register_buffer("joint_mirror_perm", torch.tensor(joint_perm, dtype=torch.long), persistent=False)
            self.register_buffer("joint_mirror_sign", torch.tensor(joint_sign, dtype=torch.float32), persistent=False)
            self.register_buffer("feet_name_mirror_perm", torch.tensor(feet_name_perm, dtype=torch.long), persistent=False)
            self.register_buffer("feet_body_mirror_perm", torch.tensor(feet_body_perm, dtype=torch.long), persistent=False)

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

    def _student_latent(self, history_obs):
        if not self.equivariant:
            return self.adaptation_module(history_obs)
        mirrored_history_obs = self._mirror_history_obs(history_obs)
        return 0.5 * (self.adaptation_module(history_obs) + self.adaptation_module(mirrored_history_obs))

    def _mirror_history_obs(self, history_obs):
        if self.obs_history_length <= 0 or self.history_input_dim % self.obs_history_length != 0:
            raise ValueError("RLvRL history observation dim must be divisible by obs_history_length.")
        history_step_dim = self.history_input_dim // self.obs_history_length
        mirrored = self._mirror_base_obs(
            history_obs.reshape(*history_obs.shape[:-1], self.obs_history_length, history_step_dim)
        )
        return mirrored.reshape_as(history_obs)

    def forward(self, obs, history_obs):
        latent = self._student_latent(history_obs)
        actor_input = torch.cat((obs, latent), dim=-1)
        mean = self.actor_body(actor_input)
        if not self.equivariant:
            return mean
        mirrored_obs = self._mirror_obs(obs, "policy")
        mirrored_actor_input = torch.cat((mirrored_obs, latent), dim=-1)
        mirrored_mean = self._mirror_joint_block(self.actor_body(mirrored_actor_input))
        return 0.5 * (mean + mirrored_mean)


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
    processed_action = action_center + action_scale * actor_action.
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
        agent_cfg_path = self.run_dir / "params" / "agent.yaml"
        if agent_cfg_path.exists():
            with open(agent_cfg_path, "r") as file:
                self.agent_cfg = yaml.unsafe_load(file)
        else:
            self.agent_cfg = {}

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
        self.posture_guide_arch_enabled = bool(self.cfg.get("posture_guide_arch_enabled", True))
        self.posture_guide_action_enabled = bool(self.cfg.get("posture_guide_action_enabled", True))
        self.posture_guide_observation_enabled = bool(self.cfg.get("posture_guide_observation_enabled", True))
        self.posture_guide_target_enabled = bool(self.cfg.get("posture_guide_target_enabled", True))
        self.guide_obs_dim = _infer_guide_obs_dim(self.cfg, self.base_obs_dim, self.adaptation_obs_dim)
        self.posture_joint_target_mode = str(
            os.environ.get("GO2_POSTURE_JOINT_TARGET_MODE", self.cfg.get("posture_joint_target_mode", "heuristic"))
        ).lower()
        if self.posture_joint_target_mode not in {"ik", "heuristic"}:
            raise ValueError("posture_joint_target_mode must be 'ik' or 'heuristic'")
        self.use_rlvrl_teacher_student = bool(self.cfg.get("use_rlvrl_teacher_student", False))
        self.rlvrl_adaptation_only = bool(self.cfg.get("rlvrl_adaptation_only", False))
        self.RL_FREQ = 1.0 / (float(self.cfg["sim"]["dt"]) * float(self.cfg["decimation"]))
        self.joint_order = os.environ.get("GO2_POSTURE_JOINT_ORDER", "joint_grouped").strip().lower()
        if self.joint_order not in {"leg_grouped", "joint_grouped"}:
            raise ValueError("GO2_POSTURE_JOINT_ORDER must be 'leg_grouped' or 'joint_grouped'")
        policy_cfg = self.agent_cfg.get("policy", {}) if isinstance(self.agent_cfg, dict) else {}
        self.policy_class_name = str(
            policy_cfg.get(
                "class_name",
                "RLvRLActorCritic" if self.use_rlvrl_teacher_student else "ActorCritic",
            )
        )
        self.use_equivariant_actor = self.policy_class_name in {
            "Go2EquivariantActorInvariantCritic",
            "RLvRLEquivariantActorInvariantCritic",
        }
        self.policy_activation = str(policy_cfg.get("activation", "elu"))

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"]
        if self.use_rlvrl_teacher_student:
            self.actor = _RLvRLActorMLP(
                state_dict,
                actor_obs_dim=int(self.cfg["rlvrl_actor_observation_dim"]),
                history_obs_dim=int(self.cfg["rlvrl_history_observation_dim"]),
                latent_dim=int(self.cfg["rlvrl_latent_dim"]),
                activation=self.policy_activation,
                equivariant=self.use_equivariant_actor,
                cfg=self.cfg,
                joint_order=self.joint_order,
                policy_cfg=policy_cfg,
            ).to(self.device)
        elif self.use_equivariant_actor:
            self.actor = _Go2EquivariantActorMLP(
                state_dict,
                cfg=self.cfg,
                joint_order=self.joint_order,
                policy_cfg=policy_cfg,
            ).to(self.device)
        else:
            self.actor = _ActorMLP(state_dict, activation=self.policy_activation).to(self.device)
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
        self.guide_lateral_shift = 0.0
        self.guide_action = self.default_joint_pos.copy()
        self.guide_action_policy = self.default_joint_pos_policy.copy()
        self.estimated_base_lin_vel = np.full(3, np.nan, dtype=np.float32)
        self.estimated_base_lin_vel_source = "uninitialized"

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
            f"joint_order={self.joint_order} policy_class={self.policy_class_name}"
            f" activation={self.policy_activation}"
            f" eqic={self.use_equivariant_actor} rlvrl={self.use_rlvrl_teacher_student}"
            f" rlvrl_actor_obs={self.rlvrl_actor_obs_mode}"
            f" rlvrl_adaptation_only={self.rlvrl_adaptation_only}"
            f" concurrent_mode={self.concurrent_state_estimator_mode}"
            f" posture_joint_target_mode={self.posture_joint_target_mode}"
            f" guide_obs_dim={self.guide_obs_dim}"
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

        prev_guide_obs = self._get_guide_observation()
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
        adaptation_obs_flat = None if self.rlvrl_adaptation_only else self._append_adaptation_history(adaptation_obs)

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
                estimated_velocity_source = "cse"
            else:
                explicit_state_obs = self._oracle_concurrent_explicit_state(base_lin_vel, feet_contacts, base_pos)
                estimated_velocity_source = "oracle"
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
            estimated_velocity_source = "cse"
            explicit_state_obs = None
            estimator_obs = None
            estimator_obs_flat = None
        else:
            root_lin_vel_obs = base_lin_vel
            estimated_velocity_source = "measured"
            explicit_state_obs = None
            estimator_obs = None
            estimator_obs_flat = None

        self.estimated_base_lin_vel = np.asarray(root_lin_vel_obs, dtype=np.float32).reshape(3).copy()
        self.estimated_base_lin_vel_source = estimated_velocity_source

        self._update_guide(command, root_lin_vel_obs[0], velocity_b=root_lin_vel_obs, omega_b=base_ang_vel)
        guide_obs = self._get_guide_observation()

        if self.rlvrl_adaptation_only:
            obs_step = np.concatenate(
                [
                    base_ang_vel,
                    self._get_projected_gravity(base_quat_wxyz),
                    command,
                    joint_pos - self.default_joint_pos_policy,
                    joint_vel,
                    self.last_action,
                    guide_obs,
                ]
            ).astype(np.float32)
            adaptation_obs_flat = self._append_adaptation_history(obs_step)
            obs = obs_step.copy()
        elif self.concurrent_state_estimator_mode == "explicit":
            actor_base_obs = estimator_obs_flat if self.concurrent_policy_obs_mode == "history" else estimator_obs
            obs_step = np.concatenate([actor_base_obs, explicit_state_obs, guide_obs]).astype(np.float32)
            obs = obs_step.copy()
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
            obs_history_flat = self._append_obs_history(obs_step)
            obs = obs_history_flat
        if obs_step.shape[0] != self.base_obs_dim:
            raise ValueError(f"go2_posture obs_step dim mismatch: got={obs_step.shape[0]} expected={self.base_obs_dim}")

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
        action_center_policy = (
            self.guide_action_policy if self.posture_guide_action_enabled else self.default_joint_pos_policy
        )
        joint_target_policy = action_center_policy + self.action_scale * action
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

    def _get_guide_observation(self):
        if not self.posture_guide_arch_enabled or self.guide_obs_dim == 0:
            return np.zeros(0, dtype=np.float32)
        if self.posture_guide_observation_enabled:
            if self.guide_obs_dim == 1:
                return np.array([self.guide_roll], dtype=np.float32)
            if self.guide_obs_dim == 3:
                return np.array([self.guide_roll, self.guide_pitch, self.guide_height], dtype=np.float32)
        if self.guide_obs_dim == 1:
            return np.zeros(1, dtype=np.float32)
        if self.guide_obs_dim == 3:
            return np.array([0.0, 0.0, float(self.cfg["guide_h_nom"])], dtype=np.float32)
        raise ValueError(f"Unsupported guide observation dim: {self.guide_obs_dim}")

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

        roll_sign_mode = self.cfg.get("guide_roll_sign_mode")
        if roll_sign_mode == "signed_lateral_accel":
            roll_lateral_accel = float(command[0]) * float(command[2])
            roll_signed = -float(self.cfg["guide_k_roll"]) * np.arctan(roll_lateral_accel / g)
            if abs(roll_lateral_accel) < 1e-6:
                roll_signed = 0.0
        elif roll_sign_mode == "legacy_yaw_sign":
            roll_lateral_accel = float(command[0]) * float(command[2])
            roll_raw = float(self.cfg["guide_k_roll"]) * np.arctan(abs(roll_lateral_accel) / g)
            roll_signed = -roll_raw if float(command[2]) > 0.0 else roll_raw
            if abs(float(command[2])) < 1e-6 or abs(roll_lateral_accel) < 1e-6:
                roll_signed = 0.0
        elif roll_sign_mode is None:
            # Preserve checkpoints saved before guide_roll_sign_mode was introduced.
            roll_raw = float(self.cfg["guide_k_roll"]) * np.arctan(a_lat / g)
            turn_sign = float(command[2]) if abs(float(command[2])) > 1e-6 else yaw_rate
            roll_signed = -roll_raw if turn_sign > 0.0 else roll_raw
            if abs_wz < 1e-6 or vel_mag < 1e-6:
                roll_signed = 0.0
        else:
            raise ValueError(
                "guide_roll_sign_mode must be 'signed_lateral_accel' or "
                f"'legacy_yaw_sign', got {roll_sign_mode!r}"
            )
        if bool(self.cfg.get("guide_roll_disable_for_reverse", False)) and float(command[0]) < 0.0:
            roll_signed = 0.0
        roll_clamp_enabled = bool(
            self.cfg.get("guide_roll_clamp_enabled", self.cfg.get("guide_clamp_enabled", False))
        )
        if roll_clamp_enabled:
            roll_signed = float(np.clip(roll_signed, -float(self.cfg["guide_roll_max"]), float(self.cfg["guide_roll_max"])))

        pitch = 0.0
        if not bool(self.cfg.get("height_guide_enabled", False)):
            height = float(self.cfg["guide_h_nom"])
        elif "adaptive_height_use_paper" in self.cfg:
            v_max = max(float(self.cfg.get("adaptive_height_v_max", 3.5)), 1.0e-6)
            yaw_max = max(float(self.cfg.get("adaptive_height_yaw_max", 3.0)), 1.0e-6)
            speed_weight = float(self.cfg.get("adaptive_height_speed_weight", 0.0))
            yaw_weight = float(self.cfg.get("adaptive_height_yaw_weight", 1.0))
            gamma = float(self.cfg.get("adaptive_height_gamma", 0.05))
            command_speed = float(np.linalg.norm(command[:2]))
            linear_ratio = float(np.clip(command_speed / v_max, 0.0, 1.0))
            yaw_for_height = float(command[2]) if bool(self.cfg.get("adaptive_height_use_command_yaw", True)) else yaw_rate
            yaw_ratio = float(np.clip(abs(yaw_for_height) / yaw_max, 0.0, 1.0))
            motion_ratio = float(
                np.clip(np.sqrt((speed_weight * linear_ratio) ** 2 + (yaw_weight * yaw_ratio) ** 2), 0.0, 1.0)
            )
            if bool(self.cfg.get("adaptive_height_use_paper", True)):
                height = (1.0 - gamma * motion_ratio) * float(self.cfg["guide_h_nom"])
            else:
                speed_ref = max(
                    float(self.cfg.get("guide_height_speed_gate_v_ref", self.cfg.get("adaptive_height_v_max", 3.0))),
                    1.0e-6,
                )
                speed_gate = float(np.clip(vel_mag / speed_ref, 0.0, 1.0))
                height = float(self.cfg["guide_h_nom"]) - float(self.cfg.get("guide_kh_yaw", 0.0)) * abs(yaw_for_height) * speed_gate
        else:
            pitch_raw = float(self.cfg.get("guide_k_pitch", 0.0)) * np.arctan(self.a_long_filtered / g)
            pitch = -pitch_raw
            height = (
                float(self.cfg["guide_h_nom"])
                - float(self.cfg["guide_kh_lat"]) * a_lat
                - float(self.cfg["guide_kh_long"]) * abs(self.a_long_filtered)
                - float(self.cfg["guide_kh_speed"]) * vel_mag**2
            )
        if bool(self.cfg.get("guide_clamp_enabled", True)):
            pitch = float(np.clip(pitch, -float(self.cfg["guide_pitch_max"]), float(self.cfg["guide_pitch_max"])))
            height = float(np.clip(height, float(self.cfg["guide_h_min"]), float(self.cfg["guide_h_nom"])))

        if not self.posture_guide_target_enabled:
            roll_signed = 0.0
            pitch = 0.0
            height = float(self.cfg["guide_h_nom"])
            lateral_shift = 0.0
        else:
            if bool(self.cfg.get("guide_roll_rate_limit_enabled", False)):
                delta_max = float(self.cfg.get("guide_roll_rate_limit", 1.5)) * ctrl_dt
                roll_delta = float(np.clip(roll_signed - self.guide_roll, -delta_max, delta_max))
                roll_signed = self.guide_roll + roll_delta
            lateral_shift = self._compute_lateral_shift_guide(command)

        self.guide_roll = roll_signed
        self.guide_pitch = pitch
        self.guide_height = height
        self.guide_lateral_shift = lateral_shift
        self.guide_action = self._posture_to_joint_targets(roll_signed, pitch, height, lateral_shift)
        self.guide_action_policy = self._to_policy_order(self.guide_action)

    def _compute_lateral_shift_guide(self, command):
        if not bool(self.cfg.get("lateral_shift_guide_enabled", True)):
            return 0.0
        shift = (
            float(self.cfg.get("guide_lateral_shift_gain", 0.30))
            * float(self.cfg.get("guide_h_nom", 0.34))
            * float(command[0])
            * float(command[2])
            / max(float(self.cfg.get("guide_gravity", 9.81)), 1.0e-6)
        )
        shift_max = float(self.cfg.get("guide_lateral_shift_max", 0.04))
        if shift_max > 0.0:
            shift = float(np.clip(shift, -shift_max, shift_max))
        return shift

    def _posture_to_joint_targets(self, roll, pitch, height, lateral_shift=0.0):
        if self.posture_joint_target_mode == "ik":
            return _posture_to_joint_targets_ik(
                roll,
                pitch,
                height,
                self.default_joint_pos,
                float(self.cfg["guide_h_nom"]),
                lateral_shift=lateral_shift,
            )

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
