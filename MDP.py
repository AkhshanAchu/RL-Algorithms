from utils.environments import Environment_prob
import random 
import numpy as np


class MDP:
    def __init__(self,start,gamma,env):
        self.start = start
        self.history = []
        self.done = False
        self.max_steps = 100
        self.ways =  ['N', 'S', 'E', 'W']
        self.gamma = gamma
        self.count = 0
        self.env = env
    
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
        
        self.update_episode(states, rewards, steps)
        return states, rewards, steps
    
    def discounted_return(self, rewards):
        terms = [(self.gamma**t)*r for t,r in enumerate(rewards)]
        return sum(terms), terms

    def update_episode(self,states, rewards, steps):
        g0, values = self.discounted_return(rewards)
        self.history.append((self.count,rewards,steps,states, g0, values))
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


env = Environment_prob({(0, 3): 10, (1, 3): -10}, gamma=0.9)
player = MDP((0,0),0.9,env)
for i in range(5):
    player.episode()

player.print_stats()