import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm


class bandit:
    def __init__(self,prob):
        self.prob = prob
        self.reward = 0
        self.count = 0

    def update(self):
        won = 0
        if np.random.random()>self.prob:
            won = 1
        self.count+=1
        self.reward = self.reward + (1/self.count)*(won - self.reward)
        return won


def reward_extract(bandits):
    return [i.reward for i in bandits]

episodes = 1000
epsilon = 0.2
total_win = []
chosen = []

choices = {"explore" : 0, "exploit" : 0}


prob_dist = [0.2,0.35,0.68,0.87]
bandits = [bandit(i) for i in prob_dist]


best_initialize = np.argmax(prob_dist)
model = bandits[best_initialize]
win = model.update()
total_win.append(win)


for i in tqdm(range(1,episodes)):
    best = reward_extract(bandits)
    if np.random.random() < epsilon:
        choice = np.random.randint(len(bandits))
        model = bandits[choice]
        choices["explore"] += 1
    else:
        choice = np.argmax(best)
        model = bandits[choice]
        choices["exploit"] += 1
        
    win = model.update()
    total_win.append(win)
    chosen.append(choice)

cumulative_win_rate = np.cumsum(total_win) / np.arange(1, len(total_win)+1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(total_win))
plt.xlabel("Bandit Index")
plt.ylabel("Number of Times Chosen")
plt.title("Distribution of Bandit Choices")
plt.xticks(range(len(bandits)))



plt.subplot(1, 2, 2)
plt.plot(cumulative_win_rate)
plt.xlabel("Episode")
plt.ylabel("Cumulative Win Rate")
plt.title("Performance Over Time")

plt.tight_layout()
plt.show()

print(choices)
    