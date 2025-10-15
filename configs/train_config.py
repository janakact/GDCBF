from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='safe-cbf',
        seed=-1,
        max_steps=1000_000,
        eval_episodes=20,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1000,
        eval_interval=25000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=1,
        pr_data='data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
    )

    possible_structures = {
        "fisor": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="FISOR",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="feasibility", 
                    sampling_method="ddpm", 
                    extract_method="minqc", 
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "gdcbf": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    # --- add to config ---
                    cbf_gamma = 0.99,
                    cbf_expectile_tau = 0.02,
                    cbf_admissibility_coef = 1e-3,
                    # safe_reward_mode = "piecewise",   # or "penalty"
                    unsafe_penalty_alpha = 1.0,
                    r_min = -1.0,
                    mask_unsafe_for_actor = False,


                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16, # samples per observation
                    M=0,# how many times the last step of the diffusion sampling process should be repeated
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='gaussian',#[gaussian, ln_resnet,mlp]
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.8,
                    critic_type="hj", #[hj, qc] #qc = Q-critic, which is standard in reinforcement learning for estimating action values.
                    cost_ub=150,
                    beta_schedule='linear',#[cosine, linear, vp]
                    actor_objective="feasibility",#[bc,feasibility] 
                    sampling_method="dpm_solver-1", #[dpmm_solver-1,  ddpm]
                    extract_method="maxq",#[minqc, maxq]
                    max_weight = 100.0,
                    qh_penalty_scale = 1.0
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
