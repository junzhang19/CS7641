# -*- coding: utf-8 -*-
"""
FiremanagementEnv is adapted from: http://sawcordwell.github.io/mdp/conservation/2015/01/10/possingham1997-1/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mdptoolbox import mdp
from QLearningSolver import QLearning

class FiremanagementEnv(object):
    def __init__(self):
        # The number of population abundance classes
        self.POPULATION_CLASSES = 10
        # The number of years since a fire classes
        self.FIRE_CLASSES = 15
        # The number of states
        self.STATES = self.POPULATION_CLASSES * self.FIRE_CLASSES
        # The number of actions
        self.ACTIONS = 2
        self.ACTION_NOTHING = 0
        self.ACTION_BURN = 1
    
    def check_action(self,x):
        """Check that the action is in the valid range."""
        if not (0 <= x < self.ACTIONS):
            msg = "Invalid action '%s', it should be in {0, 1}." % str(x)
            raise ValueError(msg)
            
    def check_population_class(self,x):
        """Check that the population abundance class is in the valid range."""
        if not (0 <= x < self.POPULATION_CLASSES):
            msg = "Invalid population class '%s', it should be in {0, 1, …, %d}." \
            % (str(x), self.POPULATION_CLASSES - 1)
            raise ValueError(msg)

    def check_fire_class(self,x):
        """Check that the time in years since last fire is in the valid range."""
        if not (0 <= x < self.FIRE_CLASSES):
            msg = "Invalid fire class '%s', it should be in {0, 1, …, %d}." % \
            (str(x), self.FIRE_CLASSES - 1)
            raise ValueError(msg)
    
    def check_probability(self,x, name="probability"):
        """Check that a probability is between 0 and 1."""
        if not (0 <= x <= 1):
            msg = "Invalid %s '%s', it must be in [0, 1]." % (name, str(x))
            raise ValueError(msg)
        
    def get_habitat_suitability(self, years):
        if years < 0:
            msg = "Invalid years '%s', it should be positive." % str(years)
            raise ValueError(msg)
        if years <= 5:
            return 0.2*years
        elif 5 <= years <= 10:
            return -0.1*years + 1.5
        else:
            return 0.5

    def convert_state_to_index(self, population, fire):
        self.check_population_class(population)
        self.check_fire_class(fire)
        return population*self.FIRE_CLASSES + fire

    def convert_index_to_state(self, index):
        if not (0 <= index < self.STATES):
            msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
              (str(index), self.STATES - 1)
            raise ValueError(msg)
        population = index // self.FIRE_CLASSES
        fire = index % self.FIRE_CLASSES
        return (population, fire)

    def transition_fire_state(self, F, a):
        ## Efect of action on time in years since fire.
        if a == self.ACTION_NOTHING:
            # Increase the time since the patch has been burned by one year.
            # The years since fire in patch is absorbed into the last class
            if F < self.FIRE_CLASSES - 1:
                F += 1
        elif a == self.ACTION_BURN:
            # When the patch is burned set the years since fire to 0.
            F = 0
        return F

    def get_transition_probabilities(self, s, x, F, a):
        """Calculate the transition probabilities for the given state and action.
    
        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.
        x : int
            The population abundance class of the threatened species.
        F : int
            The time in years since last fire.
        a : int
            The action undertaken.
    
        Returns
        -------
        prob : array
            The transition probabilities as a vector from state (``x``, ``F``) to
            every other state given that action ``a`` is taken.
    
        """
        # Check that input is in range
        self.check_probability(s)
        self.check_population_class(x)
        self.check_fire_class(F)
        self.check_action(a)
    
        # a vector to store the transition probabilities
        prob = np.zeros(self.STATES)
    
        # the habitat suitability value
        r = self.get_habitat_suitability(F)
        F = self.transition_fire_state(F, a)
    
        ## Population transitions
        if x == 0:
            # population abundance class stays at 0 (extinct)
            new_state = self.convert_state_to_index(0, F)
            prob[new_state] = 1
        elif x == self.POPULATION_CLASSES - 1:
            # Population abundance class either stays at maximum or transitions
            # down
            transition_same = x
            transition_down = x - 1
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == self.ACTION_BURN:
                transition_same -= 1
                transition_down -= 1
            # transition probability that abundance stays the same
            new_state = self.convert_state_to_index(transition_same, F)
            prob[new_state] = 1 - (1 - s)*(1 - r)
            # transition probability that abundance goes down
            new_state = self.convert_state_to_index(transition_down, F)
            prob[new_state] = (1 - s)*(1 - r)
        else:
            # Population abundance class can stay the same, transition up, or
            # transition down.
            transition_same = x
            transition_up = x + 1
            transition_down = x - 1
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == self.ACTION_BURN:
                transition_same -= 1
                transition_up -= 1
                # Ensure that the abundance class doesn't go to -1
                if transition_down > 0:
                    transition_down -= 1
            # transition probability that abundance stays the same
            new_state = self.convert_state_to_index(transition_same, F)
            prob[new_state] = s
            # transition probability that abundance goes up
            new_state = self.convert_state_to_index(transition_up, F)
            prob[new_state] = (1 - s)*r
            # transition probability that abundance goes down
            new_state = self.convert_state_to_index(transition_down, F)
            # In the case when transition_down = 0 before the effect of an action
            # is applied, then the final state is going to be the same as that for
            # transition_same, so we need to add the probabilities together.
            prob[new_state] += (1 - s)*(1 - r)
    
        # Make sure that the probabilities sum to one
        assert (prob.sum() - 1) < np.spacing(1)
        return prob
    
    def get_transition_and_reward_arrays(self,s):
        """Generate the fire management transition and reward matrices.
    
        The output arrays from this function are valid input to the mdptoolbox.mdp
        classes.
    
        Let ``S`` = number of states, and ``A`` = number of actions.
    
        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.
    
        Returns
        -------
        out : tuple
            ``out[0]`` contains the transition probability matrices P and
            ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
            numpy array and R is a numpy vector of length ``S``.
    
        """
        self.check_probability(s)
    
        # The transition probability array
        transition = np.zeros((self.ACTIONS, self.STATES, self.STATES))
        # The reward vector
        reward = np.zeros(self.STATES)
        # Loop over all states
        for idx in range(self.STATES):
            # Get the state index as inputs to our functions
            x, F = self.convert_index_to_state(idx)
            # The reward for being in this state is 1 if the population is extant
            if x != 0:
                reward[idx] = 1
            # Loop over all actions
            for a in range(self.ACTIONS):
                # Assign the transition probabilities for this state, action pair
                transition[a][idx] = self.get_transition_probabilities(s, x, F, a)
    
        return (transition, reward)
    
    def solveVI(self, discount, epsilon):
        T, R = self.get_transition_and_reward_arrays(0.5)
        vi = mdp.ValueIteration(T, R, discount = discount, epsilon = epsilon)
        #vi.setVerbose()
        vi.run()
        return vi
    
    def solvePI(self, discount, epsilon):
        T, R = self.get_transition_and_reward_arrays(0.5)
        pi = mdp.PolicyIterationModified(T, R, discount = discount, epsilon = epsilon)
        #pi.setVerbose()
        pi.run()
        return pi
    
    def solveQL(self, discount, n_iter, rand_ratio):
        T, R = self.get_transition_and_reward_arrays(0.5)
        np.random.seed(7641)
        qlearn = QLearning(T,R,discount = discount, n_iter = n_iter, rand_ratio = rand_ratio)
        #qlearn.setVerbose()
        qlearn.run()
        return qlearn

    def print_policy(self,policy):
        p = np.array(policy).reshape(self.POPULATION_CLASSES, self.FIRE_CLASSES)
        print("    " + " ".join("%2d" % f for f in range(self.FIRE_CLASSES)))
        print("    " + "---" * self.FIRE_CLASSES)
        for x in range(self.POPULATION_CLASSES):
            print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                         range(self.FIRE_CLASSES)))

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
    FM = FiremanagementEnv()
    
    ''' vi, pi'''
    #default setting, epsilon = 0.01, discount = 0.96
    vi = FM.solveVI(0.96,0.01)
    pi = FM.solvePI(0.96,0.01)
    print(FM.print_policy(vi.policy))
    print(FM.print_policy(pi.policy))
    plot('FM_ValueIteration_Iter_Vvar','Iterations','V-variation')
    plot('FM_PolicyIteration_Iter_Vvar','Iterations','V-variation')
    
    
    discount = np.arange(0.01,0.99,0.01)
    vi_time_d = []
    vi_iter_d = []
    pi_time_d = []
    pi_iter_d = []
    for d in discount:
        vi = FM.solveVI(d,0.01)
        pi = FM.solvePI(d,0.01)
        vi_time_d.append(vi.time)
        vi_iter_d.append(vi.iter)
        pi_time_d.append(pi.time)
        pi_iter_d.append(pi.iter)
        
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(vi_time_d)],axis=1)).to_csv('../plot_data/FM_ValueIteration_Discount_vs_Time.csv')    
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(vi_iter_d)],axis=1)).to_csv('../plot_data/FM_ValueIteration_Discount_vs_Iter.csv')
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(pi_time_d)],axis=1)).to_csv('../plot_data/FM_PolicyIteration_Discount_vs_Time.csv')
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(pi_iter_d)],axis=1)).to_csv('../plot_data/FM_PolicyIteration_Discount_vs_Iter.csv')
    plot('FM_ValueIteration_Discount_vs_Time','Discount','Run Time')
    plot('FM_ValueIteration_Discount_vs_Iter','Discount','Iterations')
    plot('FM_PolicyIteration_Discount_vs_Time','Discount','Run Time')
    plot('FM_PolicyIteration_Discount_vs_Iter','Discount','Iterations')   
    
    epsilon = np.arange(0.05,2,0.05)
    vi_time_e = []
    vi_iter_e = []
    pi_time_e = []
    pi_iter_e = []
    for e in epsilon:
        vi = FM.solveVI(0.9,e)
        pi = FM.solvePI(0.9,e)
        vi_time_e.append(vi.time)
        vi_iter_e.append(vi.iter)
        pi_time_e.append(pi.time)
        pi_iter_e.append(pi.iter)
        
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(vi_time_e)],axis=1)).to_csv('../plot_data/FM_ValueIteration_Epsilon_vs_Time.csv')    
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(vi_iter_e)],axis=1)).to_csv('../plot_data/FM_ValueIteration_Epsilon_vs_Iter.csv')
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(pi_time_e)],axis=1)).to_csv('../plot_data/FM_PolicyIteration_Epsilon_vs_Time.csv')
    pd.DataFrame(pd.concat([pd.Series(epsilon),pd.Series(pi_iter_e)],axis=1)).to_csv('../plot_data/FM_PolicyIteration_Epsilon_vs_Iter.csv')
    plot('FM_ValueIteration_Epsilon_vs_Time','Epsilon','Run Time')
    plot('FM_ValueIteration_Epsilon_vs_Iter','Epsilon','Iterations')
    plot('FM_PolicyIteration_Epsilon_vs_Time','Epsilon','Run Time')
    plot('FM_PolicyIteration_Epsilon_vs_Iter','Epsilon','Iterations')

def run_ql():
    ''' ql '''
    FM = FiremanagementEnv()
    max_iter = 10000
    discount_def = 0.96
    rand_ratio_def = 0.01
    
    #defaul setting
    ql = FM.solveQL(discount_def, n_iter = max_iter, rand_ratio = rand_ratio_def)
    print("Run Time: ", ql.time)
    print("\n")
    FM.print_policy(ql.policy)
    print("\n")
    plt.figure()
    plt.plot(range(int(1/rand_ratio_def)), ql.mean_discrepancy)
    plt.title("FM_mean_discrepancy_Inti_model")
    plt.xlabel("Number of random steps")
    plt.ylabel("Mean discrepancy")
    plt.savefig('../plots/FM_mean_discrepancy_Inti_model')
    
    #Discount
    discount = np.arange(0.01,0.96,0.05)
    ql_time_d = []
    for d in discount:
        ql = FM.solveQL(d, n_iter = max_iter, rand_ratio = rand_ratio_def)
        ql_time_d.append(ql.time)
        
    pd.DataFrame(pd.concat([pd.Series(discount),pd.Series(ql_time_d)],axis=1)).to_csv('../plot_data/FM_QL_Discount_vs_Time.csv')    
    plot('FM_QL_Discount_vs_Time','Discount','Run Time')

    #rand_ratio
    rand_ratio = np.arange(0.01, 0.1, 0.01)
    maxV = []
    for r in rand_ratio:
        ql = FM.solveQL(discount_def, n_iter = max_iter, rand_ratio = r)
        tmp_sum = 0
        for v in ql.V:
            tmp_sum =+ v
        maxV.append(tmp_sum)
        
    pd.DataFrame(pd.concat([pd.Series(rand_ratio),pd.Series(maxV)],axis=1)).to_csv('../plot_data/FM_QL_RandRatio_vs_MaxV.csv')    
    plot('FM_QL_RandRatio_vs_MaxV','Ratio of Random Steps','Sum of Max Value')
    
    index = maxV.index(max(maxV))
    #setting of maxV
    if rand_ratio[index] - rand_ratio_def > 0.001:
        ql = FM.solveQL(discount_def, n_iter = max_iter, rand_ratio = rand_ratio[index])
        print("New Rand_Ratio:", rand_ratio[index])
        print("Run Time: ", ql.time)
        print("\n")
        FM.print_policy(ql.policy)
        print("\n")
        plt.figure()
        plt.plot(range(int(1/rand_ratio[index])), ql.mean_discrepancy)
        plt.title("FM_mean_discrepancy_New_model")
        plt.xlabel("Number of random steps")
        plt.ylabel("Mean discrepancy")
        plt.savefig('../plots/FM_mean_discrepancy_New_model')
    else:
        print("Defaul model is optimal")

def main():
    run_vi_pi()    
    run_ql()

if __name__ == "__main__":
    main()