"""
Low-level networks: SharedEncoder + TaskQHead.

Faithful to the Elicit proposal. Two additions baked in from the lessons learned:
  - load_pretrained / freeze helpers, so the encoder and the K task heads can be
    your *already-trained* contribution (frozen), exactly like the greedy paper.
  - the final linear of each head is exposed as `.out` because PopArt rescales it
    (only when the head is trainable; see popart.py).
"""
from typing import Sequence
import os
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from Algorithm.RainbowDQL.Networks.network import DQFDuelingVisualNetwork
    _DQF_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    DQFDuelingVisualNetwork = None
    _DQF_IMPORT_ERROR = exc


class SharedEncoder(nn.Module):
    # Encodes one agent's flattened observation -> latent z_i. Shared across agents and heads.
    def __init__(self, obs_dim: int, d_enc: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, d_enc), nn.ReLU(),
        )

    def forward(self, obs):
        # obs: (B, obs_dim) -> (B, d_enc)
        return self.net(obs)


class TaskQHead(nn.Module):
    # One MLP head per role k. Maps the shared encoding to A action Q-values.
    # `self.out` is the layer PopArt rescales.
    def __init__(self, d_enc: int, A: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_enc, 128), nn.ReLU())
        self.out = nn.Linear(128, A)

    def forward(self, z):
        # z: (B, d_enc) -> (B, A)
        return self.out(self.trunk(z))


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def load_state_if_exists(module: nn.Module, path: str, map_location="cpu") -> bool:
    """Best-effort load. Returns True if a checkpoint was loaded.

    For the paper you will point this at your trained encoder/heads. For the toy
    smoke run there is no checkpoint, so it returns False and training proceeds
    from scratch.
    """
    import os
    if path and os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location=map_location))
        return True
    return False


class DuelingRolePopArtOutput:
    """PopArt adapter for one dueling role head.

    A dueling head computes Q(s,a) = V(s) + A(s,a) - mean_a A(s,a). PopArt's
    preserve-outputs transform therefore scales the advantage stream and scales
    plus shifts the value stream. This mirrors PopArt without pretending that
    your real DQFDuelingVisualNetwork is a toy single Linear head.
    """

    def __init__(self, value_layer: nn.Linear, advantage_layer: nn.Linear):
        self.value_layer = value_layer
        self.advantage_layer = advantage_layer

    @torch.no_grad()
    def popart_rescale(self, mu_old: float, sigma_old: float, mu_new: float, sigma_new: float):
        scale = sigma_old / sigma_new
        self.advantage_layer.weight.data.mul_(scale)
        self.advantage_layer.bias.data.mul_(scale)
        self.value_layer.weight.data.mul_(scale)
        self.value_layer.bias.data.copy_(
            (sigma_old * self.value_layer.bias.data + (mu_old - mu_new)) / sigma_new
        )


class DuelingNuQNetwork(nn.Module):
    """Your project low-level controller: one DQFDuelingVisualNetwork with two heads.

    Hierarchical DRL terminology used here:
      - RAM chooses the high-level option/role nu.
      - role 0 is cleaning, i.e. nu=0 in your wrapper.
      - role 1 is exploration, i.e. nu=1 in your wrapper.
      - this network is the low-level option policy/value function. Given the
        selected role, it returns Q-values over navigation actions.

    DQFDuelingVisualNetwork orders its two heads as:
      head 0: condition=True  -> exploration path in Expert_nu
      head 1: condition=False -> cleaning path in Expert_nu

    Therefore the default role_to_head=(1, 0) maps
      role 0 cleaning     -> network head 1
      role 1 exploration  -> network head 0
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        action_space_n: int = 16,
        movement_actions: int = 8,
        number_of_features: int = 1024,
        archtype: str = "v1",
        nettype: str = "0",
        role_to_head: Sequence[int] = (1, 0),
    ):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.action_space_n = int(action_space_n)
        self.movement_actions = int(movement_actions)
        self.out_dims = [self.movement_actions, self.action_space_n - self.movement_actions]
        if min(self.out_dims) <= 0:
            raise ValueError(f"Invalid action split for action_space_n={action_space_n}")
        if len(set(self.out_dims)) != 1:
            raise ValueError(
                "CTDE-RAM currently expects both DQFDueling heads to expose the same "
                f"navigation action count; got {self.out_dims}."
            )
        self.A = self.out_dims[0]
        self.role_to_head = tuple(int(x) for x in role_to_head)
        self.K = len(self.role_to_head)
        self.archtype = archtype

        if DQFDuelingVisualNetwork is None:
            raise ImportError(
                "DQFDuelingVisualNetwork could not be imported. Install the existing "
                "project dependencies (notably gym) before using low_level_backend='dueling_nu'."
            ) from _DQF_IMPORT_ERROR

        self.dqn = DQFDuelingVisualNetwork(
            self.obs_shape,
            self.out_dims,
            number_of_features,
            archtype,
            nettype,
        )
        # With the current DQFDuelingVisualNetwork both role feature streams end at 256.
        self.d_enc = 256 if archtype in ("v1", "v2") else 256

    def _head_slice(self, head_idx: int):
        start = int(sum(self.out_dims[:head_idx]))
        end = start + int(self.out_dims[head_idx])
        return slice(start, end)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the shared latent z_i consumed by the RAM aggregator."""
        if self.archtype == "v1":
            return self.dqn.feature_layer(obs.float())
        if self.archtype == "v2":
            base = self.dqn.feature_layer(obs.float())
            # Average both dense streams so the RAM sees one role-agnostic state.
            return 0.5 * (self.dqn.dense_layer1(base) + self.dqn.dense_layer2(base))
        raise ValueError(f"Unsupported archtype for CTDE encoding: {self.archtype}")

    def q_role(self, obs: torch.Tensor, role: int) -> torch.Tensor:
        """Q-values for one hierarchical role over navigation actions."""
        head_idx = self.role_to_head[int(role)]
        return self.dqn(obs.float())[:, self._head_slice(head_idx)]

    def role_output_adapter(self, role: int) -> DuelingRolePopArtOutput:
        """Return the PopArt adapter for one role head."""
        head_idx = self.role_to_head[int(role)]
        if head_idx == 0:
            return DuelingRolePopArtOutput(self.dqn.value_layer1, self.dqn.advantage_layer1)
        if head_idx == 1:
            return DuelingRolePopArtOutput(self.dqn.value_layer2, self.dqn.advantage_layer2)
        raise ValueError(f"Unsupported DQFDueling head index: {head_idx}")


def load_dqn_state_if_exists(module: DuelingNuQNetwork, path: str, map_location="cpu") -> bool:
    """Load a DQFDueling checkpoint into the wrapped network.

    Most existing project checkpoints are state_dicts for DQFDuelingVisualNetwork
    itself. This helper accepts either that format or a wrapper-level state_dict.
    """
    if not path or not os.path.exists(path):
        return False
    state = torch.load(path, map_location=map_location)
    try:
        module.dqn.load_state_dict(state)
    except RuntimeError:
        module.load_state_dict(state)
    return True
