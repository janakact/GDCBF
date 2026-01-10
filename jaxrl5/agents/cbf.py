import distrax
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union, Any
import flax.linen as nn
from flax.core import FrozenDict
# import gym 
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import GaussianPolicy
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue,get_weight_decay_mask,MLPResNet
from jaxrl5.networks import DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, get_weight_decay_mask, vp_beta_schedule

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values
@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class CBF(Agent):
    score_model: TrainState
    # target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    actor_tau: float
    reward_tau: float
    cost_tau: float
    # critic_objective: str = struct.field(pytree_node=False)
    # critic_type: str = struct.field(pytree_node=False)
    # actor_objective: str = struct.field(pytree_node=False)
    # sampling_method: str = struct.field(pytree_node=False)
    # extract_method: str = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
      
    # tanh_scale: float
    r_min: float 
    # mode:int = struct.field(pytree_node=False)  # 1: 'fisor', 2: 'add', 3: 'reach'

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        reward_tau: float = 0.8,
        cost_tau: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        # actor_objective: str = 'bc',
        # critic_objective: str = 'expectile',
        # critic_type: str = 'hj',
        # sampling_method: str = 'ddpm',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        # extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.,
        # tanh_scale: float = 5,
        r_min: float = -1.0,
        # mode: int = 1,  # 1: 'fisor', 2: 'add', 3: 'reach'
        critic_layer_norm: bool = False,
        cbf_lr: float = 3e-4,
    ):
        print("Wrong file")
        exit()
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]
        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(MLP, hidden_dims=(128, 128), activations=nn.relu, activate_final=False)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        
        # base_model_cls = partial(MLP, hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]), activations=nn.relu, activate_final=False)
        # actor_def = DDPM(time_preprocess_cls=preprocess_time_cls, cond_encoder_cls=cond_model_cls, reverse_encoder_cls=base_model_cls)
        
        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=actor_layer_norm,
                                    activate_final=False)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=actor_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=actor_dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions, time)['params']
        actor_params = FrozenDict(actor_params) 

        score_model = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adamw(learning_rate=actor_lr, weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0, mask=get_weight_decay_mask,))
        # target_score_model = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True,use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_params = FrozenDict(critic_params)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        # if critic_type == 'qc':
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic_params = FrozenDict(safe_critic_params)
        safe_critic_optimiser = optax.adam(learning_rate=cbf_lr)
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=safe_critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_params = FrozenDict(value_params)
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)
        # target_value = TrainState.create(
        #     apply_fn=value_def.apply,
        #     params=value_params,
        #     tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        # )
        # if critic_type == 'qc':
        value_def = StateValue(base_cls=value_base_cls)            
        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value_params = FrozenDict(safe_value_params)
        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')
        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            # target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            action_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            # actor_objective=actor_objective,
            # sampling_method=sampling_method,
            # critic_objective=critic_objective,
            # critic_type=critic_type,
            reward_tau=reward_tau,
            cost_tau=cost_tau,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            # extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            r_min=r_min,
            # mode=mode,
            # tanh_scale=tanh_scale
        )

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.action_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        
        '''
        reward adv
        '''
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        '''
        cost reward
        '''
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        qc = qcs.max(axis=0)

        vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"]
            )

        # if agent.critic_type == "qc":
        qc = qc - agent.qc_thres
        vc = vc - agent.qc_thres

        # if agent.actor_objective == "feasibility":
        eps = 0. #if agent.critic_type != 'qc' else 0.
        
        unsafe_condition = jnp.where( vc >  0. - eps, 1, 0)
        safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc<=0. - eps, 1, 0)
        
        cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
        reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
        
        unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub) ## ignore vc >0, qc>vc
        safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        
        weights = unsafe_weights + safe_weights
        
        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * weights).mean()
            
            return actor_loss, {'actor_loss': actor_loss, 'weights' : weights.mean()}
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        score_params = optax.incremental_update(
            score_model.params, agent.score_model.params, agent.actor_tau
        )

        score_model = agent.score_model.replace(params=score_params)

        new_agent = agent.replace(score_model=score_model, rng=rng)

        return new_agent, info

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)

        score_params = self.score_model.params
        actions, rng = ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.action_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        
        
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions)

        # if self.critic_type == "qc":
        qcs = qcs - self.qc_thres


        # if self.extract_method == 'maxq':
        #     idx = jnp.argmax(qs)
        # elif self.extract_method == 'minqc':
        idx = jnp.argmin(qcs)
        # else:
        #     raise ValueError(f'Invalid extract_method: {self.extract_method}')
        action = actions[idx]
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    @jax.jit
    def barrier_values(self, observations):
        return self.safe_value.apply_fn({"params": self.safe_value.params},observations)


    def update_r(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Combined reward critic and value function update."""
        
        # Step 1: Update V_r (Equation 4)
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"], batch["actions"]
        )
        q = jnp.min(qs, axis=0)
        
        def Vr_loss(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q - v, agent.reward_tau).mean()  # τ → 1
            return loss, {"v_r_loss": loss, "v_r": v.mean()}
        
        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)
        
        # Step 2: Update Q_r (Equation 3)
        qh = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch["observations"], batch["actions"]
        )
        qh_min = jnp.min(qh, axis=0)
        
        next_vr = value.apply_fn(
            {"params": value.params}, batch["next_observations"]
        )
        
        safe_mask = (qh_min <= 0)  # Q_h(s,a) ≤ 0
        unsafe_mask = (qh_min > 0)  # Q_h(s,a) > 0
        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - qh_min# * agent.qh_penalty
        # unsafe_target = - qh_min 
        target_q = safe_mask * safe_target + unsafe_mask * unsafe_target
        
        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["observations"], batch["actions"]
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean()}
        
        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)
        
        # Update target networks
        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        
        new_agent = agent.replace(
            value=value,
            critic=critic,
            target_critic=target_critic
        )
        
        return new_agent, {**vr_info, **qr_info}


    def update_h(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Combined safety critic and value function update."""
        
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)  # max over ensemble
        
        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            loss = expectile_loss(qc - vh, agent.cost_tau).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}
        
        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        next_vh = safe_value.apply_fn(
            {"params": safe_value.params}, batch["next_observations"]
        )
        
        h_sa: jax.Array = batch["costs"]
        target_qh = (1 - agent.discount) * h_sa + agent.discount * jnp.maximum(h_sa, next_vh)
        
        # if agent.mode == 1:  # FISOR
        #     target_qh = (1 - agent.discount) * h_sa + agent.discount * jnp.maximum(h_sa, next_vh)
        # elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
        #     target_qh = h_sa + agent.discount * next_vh - (1 - agent.discount) * 0.6
        # elif agent.mode == 3:  # Reachability Constrained RL
        #     target_qh = jnp.maximum(h_sa, next_vh)
        # elif agent.mode == 10:
        #     target_qh = jnp.maximum(h_sa, agent.discount * jnp.tanh(next_vh/agent.tanh_scale)*agent.tanh_scale)
        # else:
        #     raise ValueError(f"Unknown CBF mode: {agent.mode}")

        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"]
            )
            # TD
            # qh_loss = jnp.abs(qhs - target_qh).mean()
            qh_loss = ((qhs - target_qh) ** 2).mean()   
            return qh_loss, {"qh_loss": qh_loss, "q_h": qhs.mean()}
        
        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)
        
        # Update target networks
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        
        new_agent = agent.replace(
            safe_value=safe_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic
        )
        
        return new_agent, {**vh_info, **qh_info}


    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, h_info = new_agent.update_h(batch)        
        new_agent, r_info = new_agent.update_r(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        info = {
            **h_info,
            **r_info,
            **actor_info
        }
        return new_agent, info


    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
