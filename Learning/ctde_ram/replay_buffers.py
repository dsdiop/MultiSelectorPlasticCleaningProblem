"""
Replay buffers.

LowLevelReplayBuffer  -- one per head k. Per-agent transitions, reward = head-k signal.

RoleReplayBuffer      -- one, fleet-wide, one transition every T_role steps.

DEAD-PATH FIX (vs the Elicit delivery):
  Elicit stored the *already-pooled* role_state vector (numpy, detached). That meant
  GlobalAggregator produced part of role_state but received NO gradient -> it stayed at
  init. Here the role buffer stores the raw per-agent encodings `z_all` plus the extra
  mission vector. At update time the trainer recomputes g = GlobalAggregator(z_all),
  so the aggregator IS in the graph and DOES train. (z_all is detached data, which is
  correct for off-policy replay; the gradient enters through the *recomputed* g.)
"""
import numpy as np
import torch


def _torch_from_array(arr, dtype=None):
    # Use lists instead of torch.from_numpy so experiments still run in
    # environments where Torch was built against a different NumPy ABI.
    if dtype is None:
        return torch.tensor(arr.tolist())
    return torch.tensor(arr.tolist(), dtype=dtype)


class LowLevelReplayBuffer:
    def __init__(self, capacity: int, obs_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.obs_shape = tuple(obs_shape)
        self.obs = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def store(self, obs, action, reward_k, next_obs, done):
        i = self.ptr % self.capacity
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward_k
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            _torch_from_array(self.obs[idx], dtype=torch.float32),
            _torch_from_array(self.actions[idx], dtype=torch.long),
            _torch_from_array(self.rewards[idx], dtype=torch.float32),
            _torch_from_array(self.next_obs[idx], dtype=torch.float32),
            _torch_from_array(self.dones[idx], dtype=torch.float32),
        )


class RoleReplayBuffer:
    def __init__(self, capacity: int, N: int, d_enc: int, d_extra: int, K: int = 0):
        # Stores z_all (N, d_enc) + extra (d_extra) instead of a flat baked role_state.
        self.capacity = capacity
        self.N = N
        self.d_enc = d_enc
        self.d_extra = d_extra
        self.K = int(K)
        self.ptr = 0
        self.size = 0
        self.z = np.zeros((capacity, N, d_enc), dtype=np.float32)
        self.extra = np.zeros((capacity, d_extra), dtype=np.float32)
        # Store the actual role-weight matrix W that was executed.
        #   hard RAM: W is one-hot per agent, so argmax(W) recovers roles.
        #   soft RAM V2: W contains continuous softmax weights and must be
        #                replayed as-is for the soft role-value backup.
        self.role_weights = None
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.r_components = (
            np.zeros((capacity, self.K), dtype=np.float32) if self.K > 0 else None
        )
        self.next_z = np.zeros((capacity, N, d_enc), dtype=np.float32)
        self.next_extra = np.zeros((capacity, d_extra), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def _ensure_role_weight_storage(self, W):
        if self.role_weights is None:
            K = int(np.asarray(W).shape[-1])
            self.role_weights = np.zeros((self.capacity, self.N, K), dtype=np.float32)

    def _ensure_component_storage(self, r_components):
        if self.r_components is None:
            K = self.K or int(np.asarray(r_components).shape[-1])
            self.K = K
            self.r_components = np.zeros((self.capacity, K), dtype=np.float32)

    def store(
        self, z_all, extra, role_weights, R_role, next_z_all, next_extra, done,
        r_components=None,
    ):
        i = self.ptr % self.capacity
        self._ensure_role_weight_storage(role_weights)
        self.z[i] = z_all
        self.extra[i] = extra
        self.role_weights[i] = role_weights
        self.returns[i] = R_role
        if r_components is not None:
            self._ensure_component_storage(r_components)
            self.r_components[i] = r_components
        self.next_z[i] = next_z_all
        self.next_extra[i] = next_extra
        self.dones[i] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            _torch_from_array(self.z[idx], dtype=torch.float32),
            _torch_from_array(self.extra[idx], dtype=torch.float32),
            _torch_from_array(self.role_weights[idx], dtype=torch.float32),
            _torch_from_array(self.returns[idx], dtype=torch.float32),
            _torch_from_array(self.r_components[idx], dtype=torch.float32),
            _torch_from_array(self.next_z[idx], dtype=torch.float32),
            _torch_from_array(self.next_extra[idx], dtype=torch.float32),
            _torch_from_array(self.dones[idx], dtype=torch.float32),
        )

    def state_dict(self):
        return {
            "ptr": int(self.ptr),
            "size": int(self.size),
            "z": self.z.copy(),
            "extra": self.extra.copy(),
            "role_weights": None if self.role_weights is None else self.role_weights.copy(),
            "returns": self.returns.copy(),
            "r_components": None if self.r_components is None else self.r_components.copy(),
            "next_z": self.next_z.copy(),
            "next_extra": self.next_extra.copy(),
            "dones": self.dones.copy(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        n = min(int(state.get("size", 0)), self.capacity)
        self.ptr = int(state.get("ptr", n))
        self.size = n
        if n == 0:
            return
        for name in ("z", "extra", "returns", "next_z", "next_extra", "dones"):
            values = state.get(name)
            if values is not None:
                getattr(self, name)[:n] = np.asarray(values)[:n]
        role_weights = state.get("role_weights")
        if role_weights is not None:
            self._ensure_role_weight_storage(role_weights)
            self.role_weights[:n] = np.asarray(role_weights)[:n]
        components = state.get("r_components")
        if components is not None:
            self._ensure_component_storage(components)
            self.r_components[:n] = np.asarray(components)[:n]
