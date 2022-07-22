import math
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Latex

def get_reward(rew_avg) -> np.ndarray:
    # Add epsilon (sub-gaussian noise) to reward.
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
    epsilon = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + epsilon
    return reward

# UCB
def run_algo3(rew_avg, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    for i in range(num_trial):
        UCB = np.array([float('inf')] * 6)
        total_reward = np.zeros(6)
        T = np.zeros(6)
        for t in range(1, num_iter):
            ind = 0
            flag = 0
            for k, v in enumerate(UCB):
                if v > flag:
                    flag = v
                    ind = k
            X = get_reward(rew_avg)[ind]
            total_reward[ind] += X
            T[ind] += 1
            for j in range(6):
                if T[j] != 0:
                    UCB[j] = total_reward[j] / T[j] + math.sqrt(2 * math.log(1 / delta) / T[j])
            regret[i][t] = regret[i][t - 1] + max(rew_avg) - X
    return regret

if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_trial = int(1e4), 30
    delta = 1 / num_iter ** 2
    reg = run_algo3(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    iter = np.array([i for i in range(num_iter)])
    plt.plot(iter, avg_reg)
    plt.show()

# asymptotically optimal UCB
def run_algo6(rew_avg, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    for i in range(num_trial):
        UCB = np.array([float('inf')] * 6)
        total_reward = np.zeros(6)
        T = np.zeros(6)
        for t in range(1, num_iter):
            ind = 0
            flag = 0
            for k, v in enumerate(UCB):
                if v > flag:
                    flag = v
                    ind = k
            X = get_reward(rew_avg)[ind]
            total_reward[ind] += X
            T[ind] += 1
            for j in range(6):
                if T[j] != 0:
                    UCB[j] = total_reward[j] / T[j] + math.sqrt(2 * math.log(1 + t * math.log(t) ** 2) / T[j])
            regret[i][t] = regret[i][t - 1] + max(rew_avg) - X
    return regret

if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])
    num_iter, num_trial = int(1e4), 30
    reg = run_algo6(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    iter = np.array([i for i in range(num_iter)])
    plt.plot(iter, avg_reg)
    plt.show()
