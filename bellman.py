from utils.environments import Environment_Blocked
import random
import numpy as np


class Player:
    def __init__(self, start, gamma, env):
        self.start = start
        self.history = []
        self.done = False
        self.max_steps = 100
        self.ways =  ['N', 'S', 'E', 'W']
        self.gamma = gamma
        self.count = 0
        self.env = env
        
        self.values = np.ones((self.env.rows, self.env.column))
    def episode(self):
        states = [self.start]
        rewards = []
        done = False
        steps = 0
        state = self.start 
        while not done and steps < self.max_steps:
            action = random.choice(self.ways)
            new_state, reward, done = self.env.step(state, action)
            rewards.append(reward)
            states.append(new_state)
            state = new_state
            steps += 1
            self.bellman_update(state, reward, new_state)

        self.update_episode(states, rewards, steps)
        return states, rewards, steps
    
    def bellman_update(self, state, reward, next_state):
        next_state_value = self.values[next_state[0], next_state[1]]
        self.values[state[0], state[1]] = reward + self.gamma * next_state_value
    
    def discounted_return(self, rewards):
        terms = [(self.gamma**t)*r for t,r in enumerate(rewards)]
        return sum(terms), terms

    def update_episode(self, states, rewards, steps):
        g0, values = self.discounted_return(rewards)
        self.history.append((self.count, rewards, steps, states, g0, values))
        self.count += 1
    
    def print_stats(self):
        for episode_num, rewards, steps, states, g0, values in self.history:
            print(f"\n=== Episode {episode_num} ===")
            print(f"Steps: {steps}")
            print(f"States visited: {states}")
            print(f"Rewards: {rewards}")
            print(f"Total discounted return G0: {g0}")
            print("Discounted terms:")
            for t, val in enumerate(values):
                print(f"  Step {t}: {val:.6f}")
                
        print("\n=== Final Value Function ===")
        print(self.values)

terminal_states = {(0, 3): -10, (1, 3): 10, (2,3): -10}
blocked_states = [(2, 1)]
env = Environment_Blocked({(0, 3): -10, (1, 3): 10, (2,3): -10}, gamma=0.9)
player = Player(start=(0, 0), gamma=0.9, env=env)

for _ in range(10):
    player.episode()

print(player.values)