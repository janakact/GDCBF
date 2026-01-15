from typing import Tuple
import numpy as np
from dsrl.offline_env import filter_trajectory
from collections import defaultdict

def pre_process_data(
    self,
    data_dict: dict,
    outliers_percent: float = None,
    noise_scale: float = None,
    inpaint_ranges: Tuple[Tuple[float, float]] = None,
    epsilon: float = None,
    density: float = 1.0,
    cbins: int = 10,
    rbins: int = 50,
    max_npb: int = 5,
    min_npb: int = 2
):
    """
    pre-process the data to add outliers and/or gaussian noise and/or inpaint part of the data
    """

    # get trajectories
    done_idx = np.where(
        (data_dict["terminals"] == 1) | (data_dict["timeouts"] == 1)
    )[0]

    trajs, cost_returns, reward_returns = [], [], []
    for i in range(done_idx.shape[0]):
        start = 0 if i == 0 else done_idx[i - 1] + 1
        end = done_idx[i] + 1
        cost_return = np.sum(data_dict["costs"][start:end])
        reward_return = np.sum(data_dict["rewards"][start:end])
        traj = {k: data_dict[k][start:end] for k in data_dict.keys()}
        trajs.append(traj)
        cost_returns.append(cost_return)
        reward_returns.append(reward_return)

    print(
        f"before filter: traj num = {len(trajs)}, transitions num = {data_dict['observations'].shape[0]}"
    )

    if density != 1.0:
        assert density < 1.0, "density should be less than 1.0"
        cost_returns, reward_returns, trajs, indices = filter_trajectory(
            cost_returns,
            reward_returns,
            trajs,
            cost_min=self.min_episode_cost,
            cost_max=self.max_episode_cost,
            rew_min=self.min_episode_reward,
            rew_max=self.max_episode_reward,
            cost_bins=cbins,
            rew_bins=rbins,
            max_num_per_bin=max_npb,
            min_num_per_bin=min_npb
        )
    print(f"after filter: traj num = {len(trajs)}")

    n_trajs = len(trajs)
    traj_idx = np.arange(n_trajs)
    cost_returns = np.array(cost_returns)
    reward_returns = np.array(reward_returns)

    # outliers and inpaint ranges are based-on episode cost returns
    if outliers_percent is not None:
        assert self.target_cost is not None, \
        "Please set target cost using env.set_target_cost(target_cost) if you want to add outliers"
        outliers_num = np.max([int(n_trajs * outliers_percent), 1])
        mask = np.logical_and(
            cost_returns >= self.max_episode_cost / 2,
            reward_returns >= self.max_episode_reward / 2
        )
        print("Outliers:", outliers_percent, outliers_num, len(traj_idx), mask.shape, outliers_num/mask.sum(), outliers_num/len(traj_idx))
        # exit()
        outliers_idx = self.rng.choice(
            traj_idx[mask], size=outliers_num, replace=False
        )
        outliers_cost_returns = self.rng.choice(
            np.arange(int(self.target_cost)), size=outliers_num
        )
        # replace the original risky trajs with outliers
        for i, cost in zip(outliers_idx, outliers_cost_returns):
            len_traj = trajs[i]["observations"].shape[0]
            idx = self.rng.choice(np.arange(len_traj), cost, replace=False)
            trajs[i]["costs"] = np.zeros_like(trajs[i]["costs"])
            trajs[i]["costs"][idx] = 1
            trajs[i]["rewards"] = 1.5 * trajs[i]["rewards"]
            cost_returns[i] = cost
            reward_returns[i] = 1.5 * reward_returns[i]

    if inpaint_ranges is not None:
        inpainted_idx = []
        for inpaint_range in inpaint_ranges:
            pcmin, pcmax, prmin, prmax = inpaint_range
            cmask = np.logical_and(
                (self.max_episode_cost - self.min_episode_cost) * pcmin + self.min_episode_cost <= cost_returns[traj_idx], 
                 cost_returns[traj_idx] <= (self.max_episode_cost - self.min_episode_cost) * pcmax + self.min_episode_cost)
            rmin2 = np.min(reward_returns[traj_idx[cmask]])
            rmax2 = np.max(reward_returns[traj_idx[cmask]])
            rmask = np.logical_and(
                (rmax2 - rmin2) * prmin + rmin2 <= reward_returns[traj_idx],
                reward_returns[traj_idx] <= (rmax2 - rmin2) * prmax + rmin2
            )
            mask = np.logical_and(cmask, rmask)
            inpainted_idx.append(traj_idx[mask])
            traj_idx = traj_idx[np.logical_not(mask)]
        inpainted_idx = np.array(inpainted_idx)
        # check if outliers are filtered
        if outliers_percent is not None:
            for idx in outliers_idx:
                if idx not in traj_idx:
                    traj_idx = np.append(traj_idx, idx)

    if epsilon is not None:
        assert self.target_cost is not None, \
        "Please set target cost using env.set_target_cost(target_cost) if you want to change epsilon"
        # make it more difficult to train by filtering out
        # high reward trajectoris that satisfy target_cost
        if epsilon > 0:
            safe_idx = np.where(cost_returns <= self.target_cost)[0]
            ret = np.max(reward_returns[traj_idx[safe_idx]])
            mask = np.logical_and(
                reward_returns[traj_idx[safe_idx]] >= ret - epsilon,
                reward_returns[traj_idx[safe_idx]] <= ret
            )
            eps_reduce_idx = traj_idx[safe_idx[mask]]
            traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)
        # make it easier to train by filtering out
        # high reward trajectoris that violate target_cost
        if epsilon < 0:
            risk_idx = np.where(cost_returns > self.target_cost)[0]
            ret = np.max(reward_returns[traj_idx[risk_idx]])
            mask = np.logical_and(
                reward_returns[traj_idx[risk_idx]] >= ret + epsilon,
                reward_returns[traj_idx[risk_idx]] <= ret
            )
            eps_reduce_idx = traj_idx[risk_idx[mask]]
            traj_idx = np.setdiff1d(traj_idx, eps_reduce_idx)

    import matplotlib.pyplot as plt
    plt.figure()
    fontsize = 18
    plt.scatter(
        cost_returns[traj_idx],
        reward_returns[traj_idx],
        c='dodgerblue',
        label='remained data'
    )
    if inpaint_ranges is not None:
        plt.scatter(
            cost_returns[inpainted_idx],
            reward_returns[inpainted_idx],
            c='gray',
            label='inpainted data'
        )
    if outliers_percent is not None:
        plt.scatter(
            cost_returns[outliers_idx],
            reward_returns[outliers_idx],
            c='tomato',
            label="augmented outliers"
        )
    if epsilon is not None:
        plt.scatter(
            cost_returns[eps_reduce_idx],
            reward_returns[eps_reduce_idx],
            c='grey',
            label="filtered data"
        )
    if self.target_cost is not None:
        plt.axvline(self.target_cost, linestyle='--', label="cost limit")
    plt.legend(fontsize=fontsize)
    plt.xlabel("Cost return", fontsize=fontsize)
    plt.ylabel("reward return", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(
        f"outliers{outliers_percent}_noise{noise_scale}_inpaint{inpaint_ranges}_epsilon{epsilon}.png"
    )

    processed_data_dict = defaultdict(list)
    for k in data_dict.keys():
        for i in traj_idx:
            processed_data_dict[k].append(trajs[i][k])
    processed_data_dict = {
        k: np.concatenate(v)
        for k, v in processed_data_dict.items()
    }

    # perturbed observations
    if noise_scale is not None:
        noise_size = processed_data_dict["observations"].shape
        std = np.std(processed_data_dict["observations"])
        gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
        processed_data_dict["observations"] += gaussian_noise
        std = np.std(processed_data_dict["next_observations"])
        gaussian_noise = noise_scale * self.rng.normal(0, std, noise_size)
        processed_data_dict["next_observations"] += gaussian_noise

    return processed_data_dict
