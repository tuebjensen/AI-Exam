import numpy as np
import matplotlib.pyplot as plt

def get_reward(arm):
    return np.random.normal(q_star[arm], deviations[arm])

def select_arm(epsilon):
    if np.random.uniform(0, 1) <= epsilon:
        return np.random.randint(0, arms)
    return np.argmax(Q_epsilon)


arms = 10
iterations = 10000
epsilon = 0.1
q_star = np.random.uniform(0, 1, arms) #Expected (mean) value of distributions
deviations = np.random.uniform(0.1, 1, arms)

Q_epsilon = np.zeros(arms)
actions = np.zeros(arms)
#rewards = []
success = 0
for _ in range(100):
    for _ in range(iterations):
        arm = select_arm(epsilon)
        actions[arm] += 1
        reward = get_reward(arm)
        Q_epsilon[arm] = Q_epsilon[arm] + (reward - Q_epsilon[arm]) / actions[arm]
    
    if (np.argmax(q_star) == np.argmax(Q_epsilon)):
        success += 1
    
print(q_star)
#print(deviations)
#print(actions)
print(Q_epsilon)

print(success)