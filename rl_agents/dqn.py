import random, numpy as np, torch
import torch.nn as nn
from collections import deque
from .base_agent import BaseAgent

class MLP(nn.Module):
    def __init__(self,n_state,n_actions):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n_state,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x): return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self,n_state,n_actions,lr=1e-3,gamma=.99,epsilon=1.,epsilon_min=.05,decay=.995):
        self.n_actions=n_actions; self.gamma=gamma; self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.decay=decay
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=MLP(n_state,n_actions).to(self.device); self.target=MLP(n_state,n_actions).to(self.device); self.target.load_state_dict(self.net.state_dict())
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr); self.memory=deque(maxlen=10000)
    def act(self,state,training=True):
        if training and np.random.rand()<self.epsilon: return np.random.randint(self.n_actions)
        with torch.no_grad(): return int(self.net(torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)).argmax())
    def train_episode(self,env,batch_size=64):
        s,_=env.reset(); total=0; losses=[]; done=False
        while not done:
            a=self.act(s,True); ns,r,done,_,_=env.step(a); self.memory.append((s,a,r,ns,done)); s=ns; total+=r
            if len(self.memory)>=batch_size:
                batch=random.sample(self.memory,batch_size); ss,aa,rr,nn_,dd=zip(*batch)
                ss=torch.tensor(np.array(ss),dtype=torch.float32,device=self.device); nnx=torch.tensor(np.array(nn_),dtype=torch.float32,device=self.device)
                aa=torch.tensor(aa,device=self.device); rr=torch.tensor(rr,dtype=torch.float32,device=self.device); dd=torch.tensor(dd,dtype=torch.float32,device=self.device)
                q=self.net(ss).gather(1,aa[:,None]).squeeze()
                target=rr+self.gamma*(1-dd)*self.target(nnx).max(1).values.detach()
                loss=nn.functional.mse_loss(q,target); self.opt.zero_grad(); loss.backward(); self.opt.step(); losses.append(loss.item())
        self.epsilon=max(self.epsilon_min,self.epsilon*self.decay); self.target.load_state_dict(self.net.state_dict())
        return total, losses
