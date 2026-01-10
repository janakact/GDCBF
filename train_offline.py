import jax
import optax
import jax.numpy as jnp
import pickle
import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
# import gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF 
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_md, evaluate_pr #, offline_evaluation , plot_cbf_cost_vs_safe_value, calculate_coverage
import json

FLAGS = flags.FLAGS
# flags.DEFINE_string("config", "train_config.py:r", "Config file")
flags.DEFINE_integer('env_id', 23, 'Choose env')
# flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
# flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_integer('seed', 1, 'Seed')
# flags.DEFINE_integer('max_steps', 500_001, 'max steps')
# flags.DEFINE_integer('eval', 10000, 'eval steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')

config_flags.DEFINE_config_file(
    "config",
    None,
    # "configs/train_config.py",
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def call_main(details, env_id):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    # wandb.init( name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name']) #,use_render=True)
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'])#, ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    # ds.seed(details['dataset_kwargs']["seed"])
    ds.seed(details["seed"])
    # obs_mean = ds.obs_mean
    # obs_std = ds.obs_std
    # print('Dataset obs mean:', obs_mean, 'obs std:', obs_std)
    config_dict = dict(details['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    save_time, eval_num = 1, 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])     
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        # if i >= (details['max_steps']-20):
        # if i % details['eval_interval'] == 0:
    if details['env_name'] == 'PointRobot':
        eval_info = evaluate_pr(agent, env, details['eval_episodes'])
    # elif env_id >= 30:
    #     eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, eval_num, agent, env, details['eval_episodes'], render=False) #, save_video=True, )
    #         # eval_num += 1        
            # eval_info = evaluate(agent, env, details['eval_episodes'], save_video=True, render=True)
    else:
        eval_info = evaluate(details['seed'], agent, env, details['eval_episodes']) #, details['agent_kwargs']['cost_limit'])

    if details['env_name'] != 'PointRobot':
        eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
    
    # print ({f"eval/{k}": v for k, v in eval_info.items()})
    wandb.log({f"{k}": v for k, v in eval_info.items()} , step=i)        
    # wandb.log({f"{k}": v for k, v in eval_info.items()} )


def main(_):
    parameters = FLAGS.config
    env_id = FLAGS.env_id
    # parameters['agent_kwargs']['mode'] = FLAGS.mode
    parameters['seed'] = FLAGS.seed
    # mode = FLAGS.mode
    algo = 'fisor' #if mode == 1 else 'tanh'
    # algo = 'tanh'
    parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[env_id]    
    parameters['group'] = parameters['env_name']
    # parameters['experiment_name'] = str(env_id) + '_'  + str(parameters['env_name']) + '_' + str(parameters['dataset_kwargs']['seed']) #str(np.random.randint(1000))
    parameters['experiment_name'] = str(env_id) + '_' + algo + '_' + str(parameters['env_name']) + '_' + str(parameters['seed']) #str(np.random.randint(1000))
    
    if env_id >= 21:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5
    
    # print(parameters)

    # if not os.path.exists(f"./results/{parameters['env_name']}/{parameters['seed']}"):
    #     os.makedirs(f"./results/{parameters['env_name']}/{parameters['seed']}")
    # with open(f"./results/{parameters['env_name']}/{parameters['seed']}/config.json", "w") as f:
    #     json.dump(to_dict(parameters), f, indent=4)
    
    # call_main(parameters, parameters['dataset_kwargs']['env_id'])
    call_main(parameters, env_id)

if __name__ == '__main__':
    app.run(main)
