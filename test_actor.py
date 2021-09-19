# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:45:22 2021

@author: sociaNET_User
"""


from pond import Actor
import torch 

import gym 


env = gym.make('Pendulum-v0')


actor = Actor().float()
actor.load_state_dict(torch.load('best.pt'))

for _ in range(10):
    s = env.reset()
    s = torch.tensor(s).unsqueeze(0).float()
    done = False 
    
    while not done:
        s_prime, r, done, _ = env.step((actor(s).squeeze().item(), ))
        s_prime = torch.tensor(s_prime).unsqueeze(0).float()
        env.render()
        s = s_prime
        
        