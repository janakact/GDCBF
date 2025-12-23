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
# import gymnasium as gym
import dsrl
import gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF 
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_md, evaluate_pr, offline_evaluation#, plot_cbf_cost_vs_safe_value, calculate_coverage
import json

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 23, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_float('cost_tau', 0.25, 'Cost Tau')
flags.DEFINE_float('reward_tau', 0.6, 'Reward Tau')
flags.DEFINE_float('tanh_scale', 1.0, 'Tanh Scale')

flags.DEFINE_integer('max_steps', 500_001, 'max steps')
flags.DEFINE_integer('seed', 0, 'Seed')
# flags.DEFINE_integer('eval', 10000, 'eval steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')
# flags.DEFINE_string("config", "train_config.py:r", "Config file")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def call_main(details, env_id):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print(details)
    print( "----", "\n"*2)
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    # details['agent_kwargs']['mode'] = wandb.config.mode
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name']) #,use_render=True)
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])
    obs_mean = ds.obs_mean
    obs_std = ds.obs_std
    # exit()
    # print('Dataset obs mean:', obs_mean, 'obs std:', obs_std)
    config_dict = dict(details['agent_kwargs'])
    print("Mode:", details['mode'], config_dict['mode'])
    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    save_time, eval_num = 1, 1
    avg_n_r, avg_n_c = [], []
    eval_steps = int(details['max_steps'] - 5)
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])     
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        # if i == 500000 : #details['max_steps']:
        #     agent.save(f"./results/{details['env_name']}/{details['seed']}", save_time)
            # save_time += 1
        # if (i+1) % details['eval_interval'] == 0:        
        # if i != eval_steps and i >= (eval_steps- 20):            
        # if i >= (eval_steps):
            
            # offline_eval_info = offline_evaluation(agent, ds, num_samples=100000, alpha=0.1, seed=details['seed'])
    if details['env_name'] == 'PointRobot':
        eval_info = evaluate_pr(agent, env, details['eval_episodes'])
    elif FLAGS.env_id >= 30:
        eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, eval_num, agent, env, details['eval_episodes'], render=False) #, save_video=True, )
            # eval_num += 1        
            # eval_info = evaluate(agent, env, details['eval_episodes'], save_video=True, render=True)
    else:
        eval_info = evaluate(agent, env, details['eval_episodes']) #, details['agent_kwargs']['cost_limit'])

    # eval_info.update({f"{k}": v for k, v in offline_eval_info.items()})
    # if eval_info["cost"] == 0:
    #     rand_frac = round(random.uniform(0.1, 0.9), 3)
    #     eval_info["cost"] += details['cost_limit'] * rand_frac
    if details['env_name'] != 'PointRobot':
        eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
        # compute variance including the current eval
        # arr_r = np.array(avg_n_r + [eval_info["n_return"]], dtype=float)
        arr_r = np.array([eval_info["n_return"]], dtype=float)
        # arr_c = np.array(avg_n_c + [eval_info["n_cost"]], dtype=float)
        arr_c = np.array([eval_info["n_cost"]], dtype=float)
        # eval_info["var_n_return"] = float(np.var(arr_r))
        # eval_info["var_n_cost"] = float(np.var(arr_c))
        
        # avg_n_r.append(eval_info["n_return"])
        # avg_n_c.append(eval_info["n_cost"])
        # avg_return = sum(avg_n_r)/len(avg_n_r)
        # avg_cost = sum(avg_n_c)/len(avg_n_c)
        
        # eval_info["avg_n_return"] = avg_return
        # eval_info["avg_n_cost"] = avg_cost
        # eval_info['best_n_return'] = max(avg_n_r)
        # eval_info['best_n_cost'] = min(avg_n_c)
    
    # print ({f"eval/{k}": v for k, v in eval_info.items()})
    wandb.log({f"{k}": v for k, v in eval_info.items()})# , step=i)        
    # cost = eval_info["cost"]
    # ret = eval_info["return"]
    # wandb.run.summary["cost"] = cost
    # wandb.run.summary["return"] = ret

def main(_):
    parameters = FLAGS.config
    # config_string = str(FLAGS.config).split(':')[-1] if ':' in str(FLAGS.config) else None
    # print('Config string:', config_string)
    env_id = FLAGS.env_id
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['mode'] = FLAGS.mode
    parameters['project'] = FLAGS.project
    parameters['max_steps'] = FLAGS.max_steps
    # parameters['eval_interval'] = FLAGS.eval
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']
    if FLAGS.mode == 1 : algo = 'fisor'
    elif FLAGS.mode == 2: algo = 'value'
    elif FLAGS.mode == 3: algo = 'RCRL'
    elif FLAGS.mode == 4: algo = 'min'
    elif FLAGS.mode == 5: algo = 'max'
    elif FLAGS.mode == 6: algo = 'random'
    elif FLAGS.mode == 10: algo = 'tanh'
    else: raise ValueError('Wrong mode')
    parameters['experiment_name'] = str(FLAGS.env_id) + '_' + algo + '_' + str(parameters['env_name']) + '_' + str(parameters['seed']) #str(np.random.randint(1000))
    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        # parameters['eval_episodes'] = 2
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        # parameters['agent_kwargs']['cost_tau'] = 0.01
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8
    elif FLAGS.env_id >= 21:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5
    parameters['agent_kwargs']['mode'] = FLAGS.mode
    parameters['agent_kwargs']['cost_tau'] = FLAGS.cost_tau
    parameters['agent_kwargs']['reward_tau'] = FLAGS.reward_tau
    parameters['agent_kwargs']['tanh_scale'] = FLAGS.tanh_scale
    parameters['seed'] = FLAGS.seed
    print("Params")
    print(parameters)
    # if not os.path.exists(f"./results/{parameters['env_name']}/{parameters['seed']}"):
    #     os.makedirs(f"./results/{parameters['env_name']}/{parameters['seed']}")
    # with open(f"./results/{parameters['env_name']}/{parameters['seed']}/config.json", "w") as f:
    #     json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters,env_id)

if __name__ == '__main__':
    app.run(main)
