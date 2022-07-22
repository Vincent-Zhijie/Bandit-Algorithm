import math
import matplotlib.pyplot as plt
import numpy as np

def get_reward(rew_avg) -> np.ndarray:
    # Add epsilon (sub-gaussian noise) to reward.
    mean = np.zeros(rew_avg.size)
    cov = np.eye(rew_avg.size)
    epsilon = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + epsilon
    return reward

def run_algo1(rew_avg, m, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    for i in range(num_trial):
        reward = np.zeros(rew_avg.size)
        for m_val in range(m):
            reward += get_reward(rew_avg)
        reward = reward / m
        flag = 0
        ind = 0
        for k, v in enumerate(reward):
            if v > flag:
                flag = v
                ind = k
        for n in range(num_iter):
            if n <= 6 * m:
                regret[i][n] = sum((max(rew_avg) - rew_avg[j]) * (n // 6 + (n % 6 >= j + 1)) for j in range(6))
            else:
                regret[i][n] = regret[i][n-1] + max(rew_avg) - get_reward(rew_avg)[ind]
    return regret

if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])  # These
    num_iter, num_trial = int(1e4), 30
    m = [30, 66, 100, 300, 800]  # TODO: setup experiments with different m values

    # Run experiments.
    regrets = []
    for m_val in m:
        reg = run_algo1(rew_avg, m_val, num_iter, num_trial)
        avg_reg = np.mean(reg, axis=0)
        regrets.append(avg_reg)

    # Plot results.
    iter = np.array([i for i in range(num_iter)])
    plt.plot(iter, regrets[0], label = 'm = 30')
    plt.plot(iter, regrets[1], label = 'm = 66')
    plt.plot(iter, regrets[2], label = 'm = 100')
    plt.plot(iter, regrets[3], label = 'm = 300')
    plt.plot(iter, regrets[4], label = 'm = 800')
    plt.xlabel('iterations')
    plt.ylabel('cumulative regrets')
    plt.legend()
    plt.title('Cumulative Regret with ETC Bandit')
    plt.show()
