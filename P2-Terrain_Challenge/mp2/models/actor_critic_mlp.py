import torch
import torch.nn as nn
import numpy as np

class ActorCriticMLP(nn.Module):
    """Actor-Critic MLP for Mini Pupper 2 with 59-dim observations"""
    def __init__(self, obs_dim=59, action_dim=12, hidden_dims=[512, 256, 128], 
                 init_action_std=1.0):
        super().__init__()
        
        # Actor network
        actor_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        actor_layers.append(nn.Linear(in_dim, action_dim))
        
        # Store as 'actor' to match RSL RL naming
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        critic_layers.append(nn.Linear(in_dim, 1))  # Critic outputs value
        
        # Store as 'critic' to match RSL RL naming
        self.critic = nn.Sequential(*critic_layers)
        
        # Action standard deviation (log scale)
        self.std = nn.Parameter(torch.ones(action_dim) * np.log(init_action_std))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, obs):
        # Return both actor output and critic value
        return self.actor(obs), self.critic(obs)
    
    def get_action(self, obs, deterministic=True):
        """Get action with optional stochasticity"""
        mean = self.actor(obs)
        if deterministic:
            return mean
        else:
            # Sample from Gaussian distribution
            std = torch.exp(self.std)
            dist = torch.distributions.Normal(mean, std)
            return dist.sample()
    
    def get_value(self, obs):
        """Get critic value"""
        return self.critic(obs)