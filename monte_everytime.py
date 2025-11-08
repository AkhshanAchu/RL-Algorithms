
import random
import numpy as np
from collections import defaultdict
from  utils.environments import Environment_Blocked

class MonteEveryTime:
    def __init__(self, start, gamma, env):
        self.start = start
        self.gamma = gamma
        self.env = env
        self.max_steps = 100 #how many episode inn each of the episode
        self.ways = ['N', 'S', 'E', 'W']

        self.V = defaultdict(float)
        self.returns = defaultdict(list)

    def generate_episode(self):
        states, rewards = [self.start], []
        state = self.start
        done = False
        steps = 0

        while not done and steps < self.max_steps:
            action = random.choice(self.ways)
            next_state, reward, done = self.env.step(state, action)
            rewards.append(reward)
            states.append(next_state)
            state = next_state
            steps += 1

        return states, rewards

    def monte_carlo(self, n_episodes=1000):
        for episode in range(n_episodes):
            states, rewards = self.generate_episode()
            G = 0
            visited = set()
            for t in reversed(range(len(states)-1)):
                state = states[t]
                reward = rewards[t]
                G = self.gamma * G + reward
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
                visited.add(state)
            if episode%100 == 0:
                self.print_values()
        
        return self.V

    def print_values(self):  #gpt made
        rows, cols = self.env.rows, self.env.column
        grid = np.zeros((rows, cols), dtype=float)
        
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                if state in self.env.blocked_states:
                    grid[r, c] = np.nan  # blocked
                elif state in self.env.terminal:
                    grid[r, c] = self.env.terminal[state]
                else:
                    grid[r, c] = self.V.get(state, 0.0)
        
        print("\n=== Value Function ===")
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if np.isnan(val):
                    print("  XXXX  ", end=" ")  # blocked cell
                else:
                    print(f"{val:7.2f}", end=" ")
            print()




terminal_states = {(0, 3): -10, (1, 3): 10, (2,3): -10}
blocked_states = [(2, 1)]
env = Environment_Blocked({(0, 3): -10, (1, 3): 10, (2,3): -10}, gamma=0.9)
player = MonteEveryTime(start=(0,0), gamma=0.9, env=env)

V = player.monte_carlo(n_episodes=500)
player.print_values()
