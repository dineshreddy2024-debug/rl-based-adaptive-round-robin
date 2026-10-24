import numpy as np
from collections import defaultdict
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self,n_actions,alpha=.1,gamma=.95,epsilon=1.0,epsilon_min=.05,decay=.995,bins=(4,5,5,5,5,5,5,5)):
        self.n_actions=n_actions; self.alpha=alpha; self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.decay=decay
        self.bins=bins; self.q=defaultdict(lambda:np.zeros(n_actions))
    def discretize(self,s):
        s=np.asarray(s); return tuple(np.minimum(np.array(self.bins)-1,(s//1).astype(int)))
    def act(self,state,training=True):
        k=self.discretize(state)
        if training and np.random.rand()<self.epsilon: return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[k]))
    def train_episode(self,env):
        s,_=env.reset(); total=0
        done=False
        while not done:
            a=self.act(s,True); ns,r,done,_,_=env.step(a)
            self.q[self.discretize(s)][a]+=self.alpha*(r+self.gamma*np.max(self.q[self.discretize(ns)])*(not done)-self.q[self.discretize(s)][a])
            s=ns; total+=r
        self.epsilon=max(self.epsilon_min,self.epsilon*self.decay)
        return total
