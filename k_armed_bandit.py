import numpy as np
import matplotlib.pyplot as plt

def generate_normally_distributed_samples(mu, sigma, num_samples, resolution=100):
    # probability_density_function = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, resolution)
    samples = []
    for i in x:
        samples += [i] * int(num_samples * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(i - mu)**2 / (2 * sigma**2)))
    return samples

def get_reward(arm):
    return np.random.normal(q_star[arm], deviations[arm])

def select_arm(epsilon):
    if np.random.uniform(0, 1) <= epsilon:
        return np.random.randint(0, arms)
    return np.argmax(Q_t)

arms = 10
iterations = 10000
epsilon = 0.1
q_star = np.random.uniform(2, 6, arms) 
deviations = np.random.uniform(0.1, 0.5, arms)

Q_t = np.zeros(arms)
actions = np.zeros(arms)
rewards = [[] for _ in range(arms)]

for _ in range(iterations):
    arm = select_arm(epsilon)
    actions[arm] += 1
    reward = get_reward(arm)
    rewards[arm].append(reward)
    Q_t[arm] = Q_t[arm] + (reward - Q_t[arm]) / actions[arm]
    
# if (np.argmax(q_star) == np.argmax(Q_epsilon)):
#     success += 1
    
print(q_star)
#print(deviations)
print(actions)
print(Q_t)

# print(success)

fig, axs = plt.subplots()

normal_distributions = []
for i in range(arms):
    mu = q_star[i]
    sigma = deviations[i]
    normal_distributions.append(generate_normally_distributed_samples(mu, sigma, 1000, 100))
real_plots = axs.violinplot(normal_distributions)
estimate_plots = axs.violinplot([rewards[i] for i in range(arms)])
axs.legend((real_plots['bodies'][0], estimate_plots['bodies'][0]), ('real', 'estimate'))
axs.set_xticks(np.arange(1, arms + 1))
axs.set_xticklabels([f'{i}' for i in range(arms)])
axs.set_title('Violin plot of bandit distributions')
axs.set_xlabel('Bandit')
axs.set_ylabel('Reward')
plt.grid()
plt.show()