# Drop-in replacement for rsl_rl.networks.Memory (RNN-based).
# Place this file at: rsl_rl/networks/transformer_memory.py

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class TransformerMemory(nn.Module):
    """Transformer-based memory for sequential decision making.

    Features:
    - Pre-LayerNorm for stable RL gradients.
    - Sliding-window batch processing to perfectly mirror inference positional encodings.
    - Dynamic cross-episode attention masking using 'dones' (masks tensor).
    """

    def __init__(
        self,
        input_size,
        context_length=16,
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        ffn_dim=512,
    ):
        super().__init__()
        self.input_size = input_size
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_size, embed_dim)

        # Positional embedding only needs to cover the context_length
        self.pos_embedding = nn.Embedding(context_length, embed_dim)

        # norm_first=True is mandatory for stable continuous control in RL
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=False,
            dropout=0.0,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.obs_buffer = None

    @property
    def hidden_states(self):
        return self.obs_buffer

    @hidden_states.setter
    def hidden_states(self, value):
        self.obs_buffer = value

    def _causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _encode(self, obs_seq):
        seq_len = obs_seq.shape[0]
        projected = self.input_proj(obs_seq)
        positions = torch.arange(seq_len, device=obs_seq.device)
        pos_enc = self.pos_embedding(positions).unsqueeze(1)
        return projected + pos_enc

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None

        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states not passed during policy update")

            seq_len, batch_size, _ = input.shape
            ctx_len = self.context_length
            device = input.device

            # 1. Concatenate history with current trajectory
            full_seq = torch.cat([hidden_states, input], dim=0)

            # 2. Build full masks tensor to track episode resets across the boundary
            hidden_masks = torch.ones((ctx_len, batch_size, 1), device=device)
            full_masks = torch.cat([hidden_masks, masks], dim=0)

            windows = []
            window_padding_masks = []

            # 3. Construct sliding windows and pad masks
            for i in range(seq_len):
                windows.append(full_seq[i : i + ctx_len])

                # Check for episode resets within this specific window
                m_window = full_masks[i : i + ctx_len]

                # A token is valid if there are NO zeros (resets) after it in the window.
                # We compute this by looking backward (flip -> cumprod -> flip back)
                valid = torch.flip(torch.cumprod(torch.flip(m_window, dims=[0]), dim=0), dims=[0])

                # True means "pad this token" (ignore it in attention)
                pad_mask = (valid.squeeze(-1) == 0.0).transpose(0, 1)  # (batch_size, ctx_len)
                window_padding_masks.append(pad_mask)

            # 4. Reshape for batched transformer processing
            windows = torch.stack(windows, dim=1)
            windows_reshaped = windows.reshape(ctx_len, seq_len * batch_size, -1)
            padding_masks_reshaped = torch.stack(window_padding_masks, dim=0).reshape(seq_len * batch_size, ctx_len)

            # 5. Process through Transformer
            encoded = self._encode(windows_reshaped)
            causal_mask = self._causal_mask(ctx_len, device)

            out = self.transformer(
                encoded,
                mask=causal_mask,
                src_key_padding_mask=padding_masks_reshaped,
            )

            # 6. Extract the current timestep embedding and reshape back
            out_last = out[-1]
            out = out_last.reshape(seq_len, batch_size, -1)
            out = unpad_trajectories(out, masks)

        else:
            if self.obs_buffer is None:
                num_envs = input.shape[0]
                self.obs_buffer = torch.zeros(
                    self.context_length, num_envs, self.input_size,
                    device=input.device,
                )

            self.obs_buffer = torch.roll(self.obs_buffer, -1, dims=0)
            self.obs_buffer[-1] = input

            encoded = self._encode(self.obs_buffer)
            causal_mask = self._causal_mask(self.context_length, input.device)
            out = self.transformer(encoded, mask=causal_mask)

            out = out[-1:]

        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            if hidden_states is None:
                self.obs_buffer = None
            else:
                self.obs_buffer = hidden_states.clone()
        elif self.obs_buffer is not None:
            if hidden_states is None:
                self.obs_buffer[:, dones == 1, :] = 0.0
            else:
                raise NotImplementedError("Resetting done envs with custom hidden states not implemented")

    def detach_hidden_states(self, dones=None):
        if self.obs_buffer is not None:
            if dones is None:
                self.obs_buffer = self.obs_buffer.detach()
            else:
                self.obs_buffer[:, dones == 1, :] = self.obs_buffer[:, dones == 1, :].detach()
