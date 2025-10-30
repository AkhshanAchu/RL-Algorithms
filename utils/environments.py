import numpy as np
import random


class Environment_prob:
    def __init__(self, terminal, gamma, rows=3, column=4, step_penality=-1):
        self.terminal = terminal
        self.gamma = gamma
        self.rows = rows
        self.column = column
        self.step_penality = step_penality
        self.dumb_actions = {
        'N': [(-1, 0), (0, -1), (0, 1)],
        'S': [(1, 0), (0, -1), (0, 1)],
        'E': [(0, 1), (-1, 0), (1, 0)],
        'W': [(0, -1), (-1, 0), (1, 0)]
        }
        self.prob = [0.8,0.1,0.1]
        self.visualize = np.zeros((self.rows,self.column),np.int64)
        self.current_state = (0,0)
        self.history = []
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]

    
    def step(self,state, action):
        self.visualize[self.current_state[0]][self.current_state[1]] = 0
        if state in self.terminal:
            return state, 0, True
        chosen_action = self.dumb_actions[action]
        move_row, move_coln =  random.choices(chosen_action, self.prob)[0]
        next_state = (state[0]+move_row, state[1]+move_coln)

        if self.in_bounds(next_state):
            self.visualize[next_state[0]][next_state[1]] = 1
            self.current_state = next_state
            self.history.append(self.visualize.copy())
            if next_state in self.terminal:
                return next_state, self.terminal[next_state], True
            
            return next_state, self.step_penality, False
        self.visualize[state[0]][state[1]] = 1
        self.history.append(self.visualize.copy())
        return self.current_state, self.step_penality, False
    
    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.column

    def reset(self):
        self.visualize = np.zeros((self.rows,self.column),np.int64)
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]
        self.current_state = (0,0)
        return self.current_state



class Environment:
    def __init__(self, terminal, gamma, rows=4, column=4, step_penality=-1):
        self.terminal = terminal
        self.gamma = gamma
        self.rows = rows
        self.column = column
        self.step_penality = step_penality
        self.dumb_actions = {
        'N': [(-1, 0), (0, -1), (0, 1)],
        'S': [(1, 0), (0, -1), (0, 1)],
        'E': [(0, 1), (-1, 0), (1, 0)],
        'W': [(0, -1), (-1, 0), (1, 0)]
        }
        self.prob = [1,0,0]
        self.visualize = np.zeros((self.rows,self.column),np.int64)
        self.current_state = (0,0)
        self.history = []
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]

    
    def step(self,state, action):
        self.visualize[self.current_state[0]][self.current_state[1]] = 0
        if state in self.terminal:
            return state, 0, True
        chosen_action = self.dumb_actions[action]
        move_row, move_coln =  random.choices(chosen_action, self.prob)[0]
        next_state = (state[0]+move_row, state[1]+move_coln)

        if self.in_bounds(next_state):
            self.visualize[next_state[0]][next_state[1]] = 1
            self.current_state = next_state
            self.history.append(self.visualize.copy())
            if next_state in self.terminal:
                return next_state, self.terminal[next_state], True
            
            return next_state, self.step_penality, False
        self.visualize[state[0]][state[1]] = 1
        self.history.append(self.visualize.copy())
        return self.current_state, self.step_penality, False
    
    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.column

    def reset(self):
        self.visualize = np.zeros((self.rows,self.column),np.int64)
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]
        self.current_state = (0,0)
        return self.current_state

class Environment_Blocked:
    def __init__(self, terminal, gamma, blocked_states=None, rows=4, column=4, step_penality=-1):
        self.terminal = terminal
        self.gamma = gamma
        self.rows = rows
        self.column = column
        self.step_penality = step_penality
        self.blocked_states = set(blocked_states) if blocked_states else set()
    
        self.dumb_actions = {
            'N': [(-1, 0), (0, -1), (0, 1)],
            'S': [(1, 0), (0, -1), (0, 1)],
            'E': [(0, 1), (-1, 0), (1, 0)],
            'W': [(0, -1), (-1, 0), (1, 0)]
        }
        
        self.prob = [1, 0, 0]
        
        self.visualize = np.zeros((self.rows, self.column), np.int64)
        self.current_state = (0, 0)
        self.history = []
        
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]

        for blocked in self.blocked_states:
            self.visualize[blocked[0]][blocked[1]] = -1
    
    def step(self, state, action):
        if state in self.terminal:
            return state, 0, True
        
        chosen_action = self.dumb_actions[action]
        move_row, move_coln = random.choices(chosen_action, self.prob)[0]

        next_state = (state[0] + move_row, state[1] + move_coln)
        
        if self.in_bounds(next_state) and next_state not in self.blocked_states:
            self.visualize[self.current_state[0]][self.current_state[1]] = 0
            self.visualize[next_state[0]][next_state[1]] = 1
            self.current_state = next_state
            self.history.append(self.visualize.copy())
            
            if next_state in self.terminal:
                return next_state, self.terminal[next_state], True
            
            return next_state, self.step_penality, False
        else:
            self.visualize[state[0]][state[1]] = 1
            self.history.append(self.visualize.copy())
            return self.current_state, self.step_penality, False
    
    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.column
    
    def reset(self):
        self.visualize = np.zeros((self.rows, self.column), np.int64)
        for i in self.terminal.keys():
            self.visualize[i[0]][i[1]] = self.terminal[i]
        for blocked in self.blocked_states:
            self.visualize[blocked[0]][blocked[1]] = -1
        
        self.current_state = (0, 0)
        return self.current_state

