from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='231225',
        # env_id=38,
        seed=-1,
        max_steps=500_001,
        eval_episodes=20,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1_000,
        # eval_interval=250_000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25.0,
        # env_id =38,
        # seed=0,
        pr_data='jaxrl5/data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
    )


    '''
    17 - 0.75, [0.41 to 0.39]

    '''
    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=10,  # FISOR
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    reward_temperature=3.0,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    actor_tau=0.001,
                    reward_tau=0.75,
                    cost_tau=0.15,
                    cost_ub=150,
                    r_min=-0.001,
                    N=16,
                    discount=0.99,                    
                    tanh_scale=1.0,
                    vh_clip=25,
                    transition_tau=0.75,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]

'''
not working well:
38 - {0.9,0.5,0.1,0.01}, {0.5,0.4,0.1,0.01}
'''
