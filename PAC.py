import numpy as np
import math
import matplotlib.pyplot as plt

class bandit:
    def __init__(self, prob, idx):
        self.prob = prob
        self.count = 0
        self.total_reward = []
        self.mean = 0
        self.reward = 0
        self.id = idx  # for tracking

    def pull(self):
        return np.random.random() < self.prob

    def update(self, times):
        for _ in range(times):
            self.count += 1
            out = self.pull()
            self.reward += (1 / self.count) * (out - self.reward)
            self.total_reward.append(self.reward)

        self.mean = np.mean(self.total_reward)

    def __repr__(self):
        return f"Bandit {self.id} (P={self.prob:.3f}) -> Mean: {self.mean:.4f}, Wins: {np.sum(self.total_reward):.1f}"


prob = [np.random.random() for _ in range(10)]
bandits = [bandit(p, idx=i) for i, p in enumerate(prob)]

epsilon = 0.2
lamb = 0.1

all_bandits = {b.id: [] for b in bandits}
generation_counts = []


generation = 0
while len(bandits) > 1:
    total_times = int((4 / (epsilon ** 2)) * math.log10(3 / lamb))
    print(f"\nGeneration : {generation} | Pulls: {total_times} ")

    for b in bandits:
        b.update(total_times)
        all_bandits[b.id].extend(b.total_reward)  # track full reward history

    performance = [b.mean for b in bandits]
    skill = performance > np.median(performance)
    new_bandits = []

    for b, keep in zip(bandits, skill):
        if keep:
            new_bandits.append(b)
        else:
            print("Removed :", b)

    generation_counts.append(len(bandits))
    bandits = new_bandits.copy()

    epsilon *= (3 / 4)
    lamb *= (1 / 2)
    generation += 1


winner = bandits[0]
print("Winner Bandit:", winner)


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for bid, rewards in all_bandits.items():
    if len(rewards) > 0:
        plt.plot(rewards, label=f"B{bid}")
plt.title("Reward Trajectory per Bandit")
plt.xlabel("Time Steps")
plt.ylabel("Estimated Mean Reward")
plt.legend(loc="upper left")

plt.subplot(1, 2, 2)
plt.plot(generation_counts, marker='o')
plt.title("Bandits Remaining Per Generation")
plt.xlabel("Generation")
plt.ylabel("Surviving Bandits")

plt.tight_layout()
plt.show()
