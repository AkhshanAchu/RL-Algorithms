from  utils.environments import Environment_Blocked_Policy
import numpy as np
import random


class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.ways = ['N', 'S', 'E', 'W']
        
        self.values = np.zeros((self.env.rows, self.env.column))
        self.policy = np.full((self.env.rows, self.env.column), 'N', dtype=str)
        for r in range(self.env.rows):
            for c in range(self.env.column):
                self.policy[r, c] = random.choice(self.ways)
    
    def policy_evaluation(self):
        while True:
            delta = 0
            new_values = np.copy(self.values)
            
            for r in range(self.env.rows):
                for c in range(self.env.column):
                    state = (r, c)
                    if self.env.is_terminal(state) or self.env.is_blocked(state):
                        continue
                    
                    action = self.policy[r, c]
                    next_state, reward, _ = self.env.step(state, action)
                    v = reward + self.gamma * self.values[next_state[0], next_state[1]]
                    delta = max(delta, abs(v - self.values[r, c]))
                    new_values[r, c] = v
            
            self.values = new_values
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for r in range(self.env.rows):
            for c in range(self.env.column):
                state = (r, c)
                if self.env.is_terminal(state) or self.env.is_blocked(state):
                    continue
                
                old_action = self.policy[r, c]
                best_action = None
                best_value = -float('inf')
                
                for action in self.ways:
                    next_state, reward, _ = self.env.step(state, action)
                    v = reward + self.gamma * self.values[next_state[0], next_state[1]]
                    if v > best_value:
                        best_value = v
                        best_action = action
                
                self.policy[r, c] = best_action
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable

    def run_policy_iteration(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"\nPolicy Iteration {iteration}")
            self.policy_evaluation()
            stable = self.policy_improvement()
            self.display()
            if stable:
                print("\nPolicy converged.")
                break

    def display(self):
        print("\nValue Function:")
        print(self.values)
        print("\nPolicy:")
        arrow_map = {'N': '↑', 'S': '↓', 'E': '→', 'W': '←'}
        for r in range(self.env.rows):
            row = []
            for c in range(self.env.column):
                if self.env.is_blocked((r, c)):
                    row.append("██")
                elif self.env.is_terminal((r, c)):
                    row.append(f"{self.env.terminal[(r,c)]:2d}")
                else:
                    row.append(arrow_map[self.policy[r, c]])
            print(" ".join(row))
        print()

terminal_states = {(0, 3): -10, (1, 3): 10, (2, 3): -10}
blocked_states = [(2, 1)]
env = Environment_Blocked_Policy(terminal_states, gamma=0.9, blocked_states=blocked_states)
    
agent = PolicyIterationAgent(env, gamma=0.9)
agent.run_policy_iteration()
