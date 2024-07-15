# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:51:07 2021

@author: KnAsTc
"""

from BPN1 import BPN as bpn 
import numpy as np

def compare(a,b):
        if(a<b):return 1
        else :return 0
        

dataset = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
answer = np.array([0, 1, 1, 0])

size = [2, 2, 1]
count=0
big=0
t = bpn(dataset = dataset, labels = answer, sample = 4, size = size,iteration=1000)
print(t.x)
t.train()
y = t.predict(dataset)
print("\n",y) 


