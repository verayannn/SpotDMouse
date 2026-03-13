# Transformer Module Installation for rsl_rl

## Files to copy

### 1. New files (copy to rsl_rl package on docker)

```bash
# Find rsl_rl location
python -c "import rsl_rl; import os; print(os.path.dirname(rsl_rl.__file__))"

# Copy the two new files
cp transformer_memory.py <rsl_rl_path>/networks/transformer_memory.py
cp actor_critic_transformer.py <rsl_rl_path>/modules/actor_critic_transformer.py
```

### 2. Edit: rsl_rl/networks/__init__.py

Add this import:
```python
from .transformer_memory import TransformerMemory
```

### 3. Edit: rsl_rl/modules/__init__.py

Add this import:
```python
from .actor_critic_transformer import ActorCriticTransformer
```

### 4. Edit: rsl_rl/runners/on_policy_runner.py

Add to the import block at the top:
```python
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticTransformer,  # <-- add this line
    ...
)
```

### 5. Runner config (in your project)

```python
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass

@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTransformer"
    context_length: int = 16
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    ffn_dim: int = 512

@configclass
class CustomQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 50
    experiment_name = "stanford_DelayedPDActuator_Transformer"
    empirical_normalization = False
    policy = RslRlPpoActorCriticTransformerCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        context_length=16,
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        ffn_dim=512,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
