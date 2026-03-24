"""Two-layer recurrent policy stack for MP2 delay-robust locomotion.

Architecture (matches the requested design):

Layer 1: Policy (LSTM)
  Inputs: [imu_ang_vel(3), projected_gravity(3), self_actions(12)]
  Output: command sequence for servo model, shape (B, H, 12)

Layer 2: Servo Model (MLP)
  Inputs: [command sequence over H steps, real_position(t)]
  Output: predicted joint position sequence, shape (B, H, 12)

The curriculum helper increases delay and disturbance ranges over training progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class TemporalPolicyLSTM(nn.Module):
    """Layer 1 policy that converts proprioceptive history into command sequence."""

    def __init__(
        self,
        input_dim: int = 18,
        action_dim: int = 12,
        horizon: int = 5,
        hidden_size: int = 256,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.temporal_head = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.command_head = nn.Linear(hidden_size, horizon * action_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward one step.

        Args:
            x: (B, input_dim), with imu_ang_vel + projected_gravity + self_actions.
            hidden_state: optional (h, c) for online recurrent deployment.

        Returns:
            command_seq: (B, horizon, action_dim)
            next_hidden_state: tuple (h, c)
        """
        feat = self.encoder(x).unsqueeze(1)  # (B, 1, hidden)
        out, hidden_state = self.temporal_head(feat, hidden_state)
        seq = self.command_head(out[:, -1, :]).view(-1, self.horizon, self.action_dim)
        return seq, hidden_state


class ServoModelMLP(nn.Module):
    """Layer 2 stateless servo predictor.

    Predicts joint positions from a horizon of command actions + current real position.
    """

    def __init__(self, action_dim: int = 12, horizon: int = 5, hidden_dims: Sequence[int] = (256, 128)) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        in_dim = horizon * action_dim + action_dim
        out_dim = horizon * action_dim

        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last, h), nn.ELU()])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, command_seq: torch.Tensor, real_pos_t: torch.Tensor) -> torch.Tensor:
        """Args:
            command_seq: (B, horizon, action_dim)
            real_pos_t: (B, action_dim)

        Returns:
            predicted_pos_seq: (B, horizon, action_dim)
        """
        flat_cmd = command_seq.reshape(command_seq.shape[0], -1)
        x = torch.cat([flat_cmd, real_pos_t], dim=-1)
        y = self.net(x)
        return y.view(-1, self.horizon, self.action_dim)


class LSTMServoPolicyStack(nn.Module):
    """Composite module wiring the 2-layer architecture end-to-end."""

    def __init__(
        self,
        input_dim: int = 18,
        action_dim: int = 12,
        horizon: int = 5,
        policy_hidden: int = 256,
        servo_hidden: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.policy = TemporalPolicyLSTM(
            input_dim=input_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_size=policy_hidden,
        )
        self.servo = ServoModelMLP(
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=servo_hidden,
        )

    def forward(
        self,
        obs: torch.Tensor,
        real_pos_t: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        command_seq, next_hidden = self.policy(obs, hidden_state)
        pred_pos_seq = self.servo(command_seq, real_pos_t)
        return command_seq, pred_pos_seq, next_hidden


@dataclass(frozen=True)
class DelayCurriculumStage:
    """Curriculum stage for delay/domain randomization."""

    name: str
    min_delay_steps: int
    max_delay_steps: int
    action_noise_std: float
    imu_noise_std: float
    terrain_level: int


class DelayCurriculum:
    """Linear stage scheduler.

    Use this scheduler during training to gradually expose the policy to larger
    actuation/observation delays and tougher terrain.
    """

    def __init__(self, stages: Sequence[DelayCurriculumStage]) -> None:
        if not stages:
            raise ValueError("DelayCurriculum requires at least one stage")
        self.stages = list(stages)

    def stage_for_progress(self, progress_01: float) -> DelayCurriculumStage:
        """Map training progress in [0, 1] to a curriculum stage."""
        p = float(min(1.0, max(0.0, progress_01)))
        idx = min(int(p * len(self.stages)), len(self.stages) - 1)
        return self.stages[idx]


def default_delay_curriculum() -> DelayCurriculum:
    """Reasonable default progression for MP2 variable-delay training."""
    return DelayCurriculum(
        [
            DelayCurriculumStage("warmup", 0, 1, 0.00, 0.00, 0),
            DelayCurriculumStage("early_delay", 1, 2, 0.01, 0.01, 1),
            DelayCurriculumStage("mid_delay", 2, 4, 0.02, 0.02, 2),
            DelayCurriculumStage("late_delay", 3, 6, 0.03, 0.03, 3),
            DelayCurriculumStage("robustness", 4, 8, 0.05, 0.04, 4),
        ]
    )
