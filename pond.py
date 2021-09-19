# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:48:02 2021

@author: sociaNET_User
"""


import gym 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm 
import torch.optim as optim 
from collections import deque
import matplotlib.pyplot as plt 
import random 
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        self.layer1 = nn.Linear(3, 100)
        self.layer2 = nn.Linear(100, 1)
        # self.layer3 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = self.layer2(x)
        x = 2 * torch.tanh(x)
        return x 
        
       
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.layer1 = nn.Linear(4, 100)
        self.layer2 = nn.Linear(100, 1)
        # self.layer3 = nn.Linear(256, 1)
        
    def forward(self, s, a):
        
        x = F.relu(self.layer1(torch.cat((s, a), 1)))
        # x = F.relu(self.layer2(x))
        x = self.layer2(x)
        
        return x 
        



if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    EPISODES = 1000
    MEM_SIZE = 2000
    re_mem = deque(maxlen=MEM_SIZE)
    
    gamma = 0.9
    
    
    EPS =  1.0 
    START_DECAY = 1
    END_DECAY = 100 
    EPS_MIN = 0.001
    DECAY_AMT = (EPS - EPS_MIN) / (END_DECAY - START_DECAY)
    
    # TAU = 0.8
    
    RENDER_EVERY = 100 
    PLOT_EVERY = 5 
    COPY_EVERY = 1
    SAVE_EVERY = 50
    
    batch_size = 32 
    
    env = gym.make('Pendulum-v0')
    max_t = env.max_torque
    
    critic = Critic().float()
    critic_prime = Critic().float()
    critic_prime.load_state_dict(critic.state_dict())
    
    actor = Actor().float()
    actor_prime = Actor().float()
    actor_prime.load_state_dict(actor.state_dict())
        
    frame = 0 
    
    rs = []
    
    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    var = 1.0 
    min_score = -1e4 
    
    for episode in tqdm(range(EPISODES)):
        s = env.reset()
        s = torch.tensor(s).unsqueeze(0).float()
                
        done  = False
        cost = 0
        while not done:
            frame += 1
                       
            # if np.random.rand() < EPS:
            #     sel_a = np.random.uniform(-max_t, max_t)
            # else:
            
            sel_a = actor(s).squeeze().item()
            sel_a = Normal(sel_a, var).sample().clamp(-max_t, max_t).item()
            # print(sel_a)
                                             
            s_prime, r, done, _ = env.step((sel_a,))
            
            s_prime = torch.tensor(s_prime).unsqueeze(0).float()
            cost += r 
            r = (r + 8) / 8 
            
            if done:
                rs.append(cost)
                if cost > min_score:
                    min_score = cost 
                    torch.save(actor.state_dict(), f'best_{episode}_{cost}.pt')
            

            obs = (s, sel_a, r, s_prime, done)
            
            
            re_mem.append(obs)
            s = s_prime
            
                        
            if frame < MEM_SIZE : 
                continue
            
            var = max(0.999 * var, 0.01)
            
            batch = random.sample(re_mem, batch_size)
            
            s_batch = torch.empty(batch_size, 3).float()
            s_prime_batch = torch.empty(batch_size, 3).float()
            a_batch = torch.empty(batch_size, 1).float()
            
            q_target = torch.empty(batch_size, 1).float()
            r_batch = torch.empty(batch_size, 1).float()
            
            
            for cnt in range(batch_size):
                s_batch[cnt] = batch[cnt][0]
                a_batch[cnt] = batch[cnt][1]
                r_batch[cnt] = batch[cnt][2]
                s_prime_batch[cnt] = batch[cnt][3]
                
            
            qs = critic(s_batch, a_batch)
            # a_prime_batch = actor_prime(s_prime_batch)
            # qs_prime = critic_prime(s_prime_batch, a_prime_batch)
               
            with torch.no_grad():
                q_target = r_batch + gamma * critic_prime(s_prime_batch, actor_prime(s_prime_batch))
      
                            
            # q_target.detach_()
                
            critic_optim.zero_grad()
            critic_loss = F.smooth_l1_loss(qs, q_target)
            # print(f'critic loss {critic_loss.item()}')
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optim.step()    
                
            
            actor_optim.zero_grad()
            actor_loss = -critic(s_batch, actor(s_batch)).mean()          
            # print(f'actor loss {actor_loss.item()}')           
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optim.step()
            
            
            
            if frame % 200 == 0:
                critic_prime.load_state_dict(critic.state_dict())
            
            if frame % 201 == 0:
                actor_prime.load_state_dict(actor.state_dict())
            
            
            
            
            if (episode + 1) % RENDER_EVERY == 0:
                env.render()
        
        if END_DECAY >= episode+1 >= START_DECAY:
            EPS -= DECAY_AMT
            
        if (episode + 1) % PLOT_EVERY == 0:
            plt.plot(rs)
            plt.show()
            
        # if (episode + 1) % COPY_EVERY == 0:
        #     critic_prime.load_state_dict(critic.state_dict())
        #     actor_prime.load_state_dict(actor.state_dict())
            
        if (episode + 1) % SAVE_EVERY == 0:
            torch.save(actor.state_dict(), 'actor.pt')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    