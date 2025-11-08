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

def ucb_bandit(bandits, episodes=1000):
    total_win = 0
    chosen = []
    wins = []
    n_bandits = len(bandits)

    for i in range(n_bandits):
        win = bandits[i].update()
        total_win += win
        chosen.append(i)
        wins.append(win)

    for t in tqdm(range(n_bandits, episodes)):
        ucb_values = []
        for b in bandits:
            if b.count == 0:
                ucb = float('inf')
            else:
                ucb = b.reward + np.sqrt(2 * np.log(t) / b.count)
            ucb_values.append(ucb)
            
        choice = np.argmax(ucb_values)
        chosen.append(choice)
        win = bandits[choice].update()
        wins.append(win)
        total_win += win

    cumulative_win_rate = np.cumsum(wins) / np.arange(1, episodes + 1)
    return wins, cumulative_win_rate


prob_dist = [0.2,0.35,0.68,0.87]
bandits_ucb = [bandit(p) for p in prob_dist]
chosen, cumulative_win_rate = ucb_bandit(bandits_ucb, episodes=5000)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.cumsum(chosen))
plt.xlabel('Bandit Index')
plt.ylabel('Times Chosen')
plt.title('Distribution of Bandit Choices')
plt.xticks(range(len(bandits_ucb)))

plt.subplot(1, 2, 2)
plt.plot(cumulative_win_rate)
plt.xlabel('Episode')
plt.ylabel('Cumulative Win Rate')
plt.title('Performance Over Time')

plt.tight_layout()
plt.show()
