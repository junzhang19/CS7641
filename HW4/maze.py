#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:14:47 2019

The environment is adapted from: https://github.com/vicflair/cs7641/blob/master/hw4/main.py
"""
from mdptoolbox import mdp
import mdptoolbox.util as util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from QLearningSolver import QLearning

class MazeEnv(object):
    def __init__(self, maze=None, goal=None, theseus=None, minotaur=None):
        # The pre-defined maze
        if maze is None:
            self.maze = np.asarray(
                [[0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0],
                 [0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 0]])
        else:
            self.maze = maze
        self.X = self.maze.shape[0]
        self.Y = self.maze.shape[1]

        # Goal position
        if goal is None:
            self.goal = (self.X-1, self.Y-1)
        else:
            self.goal = goal

        # Theseus starting position
        if theseus is None:
            self.theseus = (1, 2)
        else:
            assert self.maze[theseus] == 0
            assert theseus != self.goal
            self.theseus = theseus

        # Minotaur starting position
        if minotaur is None:
            self.minotaur = (1, 4)
        else:
            assert self.maze[minotaur] == 0
            self.minotaur = minotaur

    def convert_state_to_index(self, pos1, pos2):
        """Convert a pair of positions (x,y) to a state index."""
        num_local_states = (self.X * self.Y)
        state1 = self.Y * pos1[0] + pos1[1]
        state2 = self.Y * pos2[0] + pos2[1]
        state_index = state1 * num_local_states + state2
        return state_index

    def transition(self):
        # Initialize transition matrices
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        matrix_size = (4, num_states, num_states) ## 4 directions: N,E,W,S
        T = np.zeros(matrix_size)

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        # For every pair of positions, get transition probabilities
        pos2 = ((theseus, minotaur) for theseus in pos for minotaur in pos)
        for theseus, minotaur in pos2:
            # Get Theseus's new positions (deterministic)
            theseus_next = self.go_next(theseus)

            # Get Minotaur's possible new positions (random)
            minotaur_next = self.go_next(minotaur)

            # Update transition probabilities for each action matrix
            current_state = self.convert_state_to_index(theseus, minotaur)
            for a in range(4):
                # Get next states
                next_states = [self.convert_state_to_index(theseus_next[a], M)
                               for M in minotaur_next]
                # Update transition probabilities
                for ns in next_states:
                    T[a, current_state, ns] += 0.25 #eqaul probability

        # "Reset" to initial state when meeting minotaur.
        initial_state = self.convert_state_to_index(self.theseus, self.minotaur)
        for p in pos:
            # All states where Theseus and minotaur are in same position
            same_pos_state = self.convert_state_to_index(p, p)

            # Reset to initial state
            T[:, same_pos_state, :] = 0
            T[:, same_pos_state, initial_state] = 1

            # All states where Theseus is at the goal
            goal_state = self.convert_state_to_index(self.goal, p)

            # Reset to initial state
            T[:, goal_state, :] = 0
            T[:, goal_state, initial_state] = 1

        # Confirm stochastic matrices
        for a in range(4):
            util.checkSquareStochastic(T[a])
        return T

    def rewards(self, reward=None):
        """Returns reward matrix."""

        if reward is None:
            # Reward for goal, penalty for loss, step penalty
            self.reward = [1, -1, -0.01]
        else:
            self.reward = reward

        # Initialize rewards matrix with step penalty
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        R = np.ones((num_states, 4)) * self.reward[2]

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
        
        # Reward for goal, game ends
        for p in pos:
            R[self.convert_state_to_index(self.goal, p), :] = self.reward[0]

        # Penalty for loss, game over
        for p in pos:
            R[self.convert_state_to_index(p, p), :] = self.reward[1]

        return R

    def go_next(self, current_pos):
        """Get result of N, E, S, W move states."""
        x = current_pos[0]
        y = current_pos[1]
        next_states = []
        # go North
        if (x > 0) and (self.maze[x - 1, y] == 0):
            next_states.append((x - 1, y))
        else:
            next_states.append((x, y))
        # go East
        if (y < self.Y - 1) and (self.maze[x, y + 1] == 0):
            next_states.append((x, y + 1))
        else:
            next_states.append((x, y))
        # go South
        if (x < self.X - 1) and (self.maze[x + 1, y] == 0):
            next_states.append((x + 1, y))
        else:
            next_states.append((x, y))
        # go West
        if (y > 0) and (self.maze[x, y - 1] == 0):
            next_states.append((x, y - 1))
        else:
            next_states.append((x, y))
        return next_states

    def convert_index_to_state(self, state_index):
        """Recover pair of positions from global state."""
        num_local_states = (self.X * self.Y)
        state2 = state_index % num_local_states
        state1 = (state_index - state2) / num_local_states

        # Convert local states to positions
        pos1 = [0, 0]
        pos1[1] = state1 % self.Y
        pos1[0] = (state1 - pos1[1]) / self.Y

        pos2 = [0, 0]
        pos2[1] = state2 % self.Y
        pos2[0] = (state2 - pos2[1]) / self.Y
        return tuple(pos1), tuple(pos2)        

    def print_policy(self,policy):
        theseus_policy = []
        num_states = self.X * self.Y
        for i in range(num_states):
            theseus_policy.append(policy[i*num_states])        
        theseus_policy = np.array(theseus_policy).reshape(self.X, self.Y)    
        return theseus_policy

def plot(title, x_label, y_label):
    df = pd.read_csv('../plot_data/'+title+'.csv')
    plt.figure()
    plt.grid
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(df.iloc[:,[1]],df.iloc[:,[2]],'o-')
    plt.grid
    plt.savefig('../plots/'+title)

def run_vi_pi():
    """Solves the Maze aka Theseus and the Minotaur MDP."""
    MZ = MazeEnv()
    T = MZ.transition()
    R = MZ.rewards()

    #intial values
    vi = mdp.ValueIteration(T, R, discount = 0.9, epsilon = 0.01)
    pi = mdp.PolicyIterationModified(T, R, discount=0.9, epsilon = 0.01)
    #vi.setVerbose()
    #pi.setVerbose()
    vi.run()
    pi.run()   
    print(MZ.print_policy(vi.policy))
    print("\n")
    print(MZ.print_policy(pi.policy))
    plot('MZ_ValueIteration_Iter_Vvar','Iterations','V-variation')
    plot('MZ_PolicyIteration_Iter_Vvar','Iterations','V-variation')
    
    #Discount
    discount = np.arange(0.01,0.99,0.01)
    vi_time_d = []
    vi_iter_d = []
    pi_time_d = []
    pi_iter_d = []
    for d in discount:
        vi = mdp.ValueIteration(T, R, discount = d, epsilon = 0.01)
        pi = mdp.PolicyIterationModified(T, R, discount=d, epsilon = 0.01)
        vi.run()
        pi.run()
        vi_time_d.append(vi.time)
        vi_iter_d.append(vi.iter)
        pi_time_d.append(pi.time)
        pi_iter_d.append(pi.iter)
        
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(vi_time_d)],axis=1)).to_csv('../plot_data/MZ_ValueIteration_Discount_vs_Time.csv')    
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(vi_iter_d)],axis=1)).to_csv('../plot_data/MZ_ValueIteration_Discount_vs_Iter.csv')
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(pi_time_d)],axis=1)).to_csv('../plot_data/MZ_PolicyIteration_Discount_vs_Time.csv')
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(pi_iter_d)],axis=1)).to_csv('../plot_data/MZ_PolicyIteration_Discount_vs_Iter.csv')
    plot('MZ_ValueIteration_Discount_vs_Time','Discount','Run Time')
    plot('MZ_ValueIteration_Discount_vs_Iter','Discount','Iterations')
    plot('MZ_PolicyIteration_Discount_vs_Time','Discount','Run Time')
    plot('MZ_PolicyIteration_Discount_vs_Iter','Discount','Iterations')   
    
    #Epsilon
    epsilon = np.arange(0.05,2,0.05)
    vi_time_e = []
    vi_iter_e = []
    pi_time_e = []
    pi_iter_e = []
    for e in epsilon:
        vi = mdp.ValueIteration(T, R, discount = 0.9, epsilon = e)
        pi = mdp.PolicyIterationModified(T, R, discount=0.9, epsilon = e)
        vi.run()
        pi.run()
        vi_time_e.append(vi.time)
        vi_iter_e.append(vi.iter)
        pi_time_e.append(pi.time)
        pi_iter_e.append(pi.iter)
        
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(vi_time_e)],axis=1)).to_csv('../plot_data/MZ_ValueIteration_Epsilon_vs_Time.csv')    
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(vi_iter_e)],axis=1)).to_csv('../plot_data/MZ_ValueIteration_Epsilon_vs_Iter.csv')
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(pi_time_e)],axis=1)).to_csv('../plot_data/MZ_PolicyIteration_Epsilon_vs_Time.csv')
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(pi_iter_e)],axis=1)).to_csv('../plot_data/MZ_PolicyIteration_Epsilon_vs_Iter.csv')
    plot('MZ_ValueIteration_Epsilon_vs_Time','Epsilon','Run Time')
    plot('MZ_ValueIteration_Epsilon_vs_Iter','Epsilon','Iterations')
    plot('MZ_PolicyIteration_Epsilon_vs_Time','Epsilon','Run Time')
    plot('MZ_PolicyIteration_Epsilon_vs_Iter','Epsilon','Iterations')


def run_ql():
    ''' ql '''
    MZ = MazeEnv()
    T = MZ.transition()
    R = MZ.rewards()
    max_iter = 10000
    discount_def = 0.9
    rand_ratio_def = 0.01
    np.random.seed(7641)
    
    #defaul setting
    ql = QLearning(T,R,discount = discount_def, n_iter = max_iter, rand_ratio = rand_ratio_def)
    ql.run()
    print("Run Time: ", ql.time)
    print("\n")
    print(MZ.print_policy(ql.policy))
    print("\n")
    plt.figure()
    plt.plot(range(int(1/rand_ratio_def)), ql.mean_discrepancy)
    plt.title("MZ_mean_discrepancy_Inti_model")
    plt.xlabel("Number of random steps")
    plt.ylabel("Mean discrepancy")
    plt.savefig('../plots/MZ_mean_discrepancy_Inti_model')
    #print(ql.V)
    
    #Discount
    discount = np.arange(0.05,0.90,0.05)
    ql_time_d = []
    for d in discount:
        ql = QLearning(T,R,discount = d, n_iter = max_iter, rand_ratio = rand_ratio_def)
        ql.run()
        ql_time_d.append(ql.time)
        
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(ql_time_d)],axis=1)).to_csv('../plot_data/MZ_QL_Discount_vs_Time.csv')    
    plot('MZ_QL_Discount_vs_Time','Discount','Run Time')

    #rand_ratio
    rand_ratio = np.arange(0.01, 0.1, 0.01)
    maxV = []
    for r in rand_ratio:
        ql = QLearning(T,R,discount = discount_def, n_iter = max_iter, rand_ratio = r)
        ql.run()
        tmp_sum = 0
        for v in ql.V:
            tmp_sum =+ v
        maxV.append(tmp_sum)
        
    pd.DataFrame(pd.concat([pd.Series(rand_ratio),pd.Series(maxV)],axis=1)).to_csv('../plot_data/MZ_QL_RandRatio_vs_MaxV.csv')    
    plot('MZ_QL_RandRatio_vs_MaxV','Ratio of Random Steps','Sum of Max Value')
    
    index = maxV.index(max(maxV))
    #setting of maxV
    if rand_ratio[index] - rand_ratio_def > 0.001:
        ql = QLearning(T,R,discount = discount_def, n_iter = max_iter, rand_ratio = rand_ratio[index])
        ql.run()
        print("New Rand_Ratio:", rand_ratio[index])
        print("Run Time: ", ql.time)
        print("\n")
        print(MZ.print_policy(ql.policy))
        print("\n")
        plt.figure()
        plt.plot(range(int(1/rand_ratio[index])), ql.mean_discrepancy)
        plt.title("MZ_mean_discrepancy_New_model")
        plt.xlabel("Number of random steps")
        plt.ylabel("Mean discrepancy")
        plt.savefig('../plots/MZ_mean_discrepancy_New_model')
    else:
        print("Defaul model is optimal")


def main():
    run_vi_pi()
    run_ql()

if __name__ == "__main__":
    main()