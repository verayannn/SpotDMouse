# rsl_rl/modules/actor_critic_transformer.py

from __future__ import annotations

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import TransformerMemory


class ActorCriticTransformer(ActorCritic):
    """Actor-critic with transformer-based temporal context."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        context_length=16,
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        ffn_dim=512,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, "
                "which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=embed_dim,
            num_critic_obs=embed_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.memory_a = TransformerMemory(
            input_size=num_actor_obs,
            context_length=context_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
        )

        self.memory_c = TransformerMemory(
            input_size=num_critic_obs,
            context_length=context_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
        )

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
