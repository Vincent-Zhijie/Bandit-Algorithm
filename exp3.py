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
    
def run_algo9(rew_avg, num_iter, num_trial) -> np.ndarray:
    regret = np.zeros((num_trial, num_iter))
    k = len(rew_avg)
    mu_star = max(rew_avg)
    for trial in range(num_trial):
        S_hat = np.zeros(k)
        P = np.zeros(k)
        sum = 0
        for t in range(num_iter):
            P = np.exp(eta * S_hat)
            p_sum = np.sum(P)
            P /= p_sum
            At = int(np.random.choice(k, 1, p=P))
            Xt = get_reward(rew_avg)[At]
            sum += mu_star - Xt
            regret[trial][t] = sum
            S_hat += 1
            S_hat[At] -= (1 - Xt) / P[At]
    return regret
  
  if __name__ == '__main__':
    # Initialize experiment parameters.
    rew_avg = np.asarray([0.6, 0.9, 0.95, 0.8, 0.7, 0.3])  
    num_iter, num_trial = int(1e4), 30
    
    eta = 0.001
    reg = run_algo9(rew_avg, num_iter, num_trial)
    avg_reg1 = np.mean(reg, axis = 0)
    
    eta = 0.002
    reg = run_algo9(rew_avg, num_iter, num_trial)
    avg_reg2 = np.mean(reg, axis = 0)
    
    eta = math.sqrt(2 * math.log(6) / (num_iter * 6))
    reg = run_algo9(rew_avg, num_iter, num_trial)
    avg_reg3 = np.mean(reg, axis = 0)

    iter = np.array([i for i in range(num_iter)])
    plt.plot(iter, avg_reg1, label = r'Exp3($\eta = 0.001$)')
    plt.plot(iter, avg_reg2, label = r'Exp3($\eta = 0.002$)')
    plt.plot(iter, avg_reg3, label = r'Exp3($\eta = \sqrt{2log(k)/(nk)}$)')
    plt.xlabel('iterations')
    plt.ylabel('cumulative regrets')
    plt.legend()
    plt.title(r'Exp3 with different $\eta$')
    plt.show()
    
    
