import gym
import numpy as np
import math

from Utils.utils import *
from Utils.img_aug_func import *

from gym.spaces import Box, Discrete, Tuple
import matplotlib.pyplot as plt
import albumentations as A

import skimage.io as io

from skimage.measure import label

from skimage.morphology import binary_dilation
from skimage.measure import regionprops

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import relabel_sequential

from scipy.ndimage import distance_transform_edt
import scipy.ndimage.filters as fi

from enum import IntEnum
import copy

import cv2

import os, time

class QueryType (IntEnum):
    Delete = 1
    Insert = 2
    Read = 3


class database_env (gym.Env):
    def __init__ (self, config, seed=0, dstype='train'):
        self.rng = np.random.RandomState (seed)
        # self.observation_space = Box (-1.0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        self.max_step = config ["max_step"]
        
        self.num_range  = 100
        self.num_subrange = 10
        self.types = list (range (3))
        self.size = 128
        self.pool_size = 15

        self.dstype=dstype

        self.key_his = [0] * self.size
        self.query_his = [0] * self.size

        self.type1_data = []
        self.type2_data = []
        self.type3_data = []

        self.type1_pool = []
        self.type2_pool = []
        self.type3_pool = []

        self.action_his = []
        self.rewards = []

        self.next_behavior = 0
        self.next_cost = 0

        self.config = config
        self.current_time = 0

    def debug_pool (self):
        print (self.type1_pool)
        print (self.type2_pool)
        print (self.type3_pool)

    def general_behaviror (self):
        next_behavior = self.rng.randint (3)

        if next_behavior == 0:
            self.query_type_1 ()
            self.next_cost = 1.0
        if next_behavior == 1:
            self.query_type_2 ()
            self.next_cost = 1.0
        if next_behavior == 2:
            self.query_type_3 ()
            self.next_cost = 1.0

        while len (self.key_his) > self.size:
            self.key_his.pop (0)
            self.query_his.pop (0)

        self.next_behavior = next_behavior

    def query_type_1 (self):
        # Repeatedly_delete_insert
        prob_list_type2 = [0.7, 0.15, 0.15]
        query_type = self.rng.choice (list (QueryType), 1, replace=True, p=prob_list_type2) [0]
        key = self.rng.choice (self.type1_pool, 1) [0]
        self.key_his.append (key)
        self.query_his.append (query_type)

    def query_type_2 (self):
        # Only read, insert
        prob_list_type2 = [0.15, 0.7, 0.15]
        query_type = self.rng.choice (list (QueryType), 1, replace=True, p=prob_list_type2) [0]
        key = self.rng.choice (self.type2_pool, 1) [0]
        self.key_his.append (key)
        self.query_his.append (query_type)

    def query_type_3 (self):
        # Sequencial access, rarely delete, insert
        l = self.rng.randint (3, 4)
        r = self.rng.randint (l, len (self.type3_pool))

        prob_list_type3 = [0.15, 0.15, 0.7]

        for i in range (l, r + 1):
            query_type = self.rng.choice (list (QueryType), 1, replace=True, p=prob_list_type3) [0]
            key = self.type3_pool [i]

            self.key_his.append (key)
            self.query_his.append (query_type)

    def pool_reset (self):
        self.type1_pool = []
        self.type2_pool = []
        self.type3_pool = []
        choices = list (range (1, 101))
        size = self.pool_size
        self.type1_pool = self.rng.choice (choices, size, replace=False)
        for x in self.type1_pool:
            choices.remove (x)
        self.type2_pool = self.rng.choice (choices, size, replace=False)
        for x in self.type2_pool:
            choices.remove (x)
        self.type3_pool = self.rng.choice (choices, size, replace=False)
        for x in self.type3_pool:
            choices.remove (x)

    def reset (self):
        self.type1_data = []
        self.type2_data = []
        self.type3_data = []

        self.action_hist = []

        self.key_his = [0] * self.size
        self.query_his = [0] * self.size
        self.pool_reset ()
        # self.random_start ()

        self.sum_reward = 0
        self.nstep = 0
        self.rewards = []
        self.old_acts = []
        return self.observation ()

    def random_start (self):
        n = self.rng.random (len (10))
        for i in range (10):
            self.general_behaviror ()

    def visualize (self):
        os.system('clear')

        print ("step: ", self.nstep)

        print ("Pool I:", end=' ')
        print (self.type1_pool, '- Reward DS_I: 1.0, DS_II: 0.0, DS_III: 0.0')
        print ("Pool II:", end=' ')
        print (self.type2_pool, '- Reward DS_I: 0.0, DS_II: 1.0, DS_III: 0.0')
        print ("Pool III:", end=' ')
        print (self.type3_pool, '- Reward DS_I: 0.0, DS_II: 0.0, DS_III: 1.0')
        print ()
        print ("History: ", end=' ')
        tmp = []
        for i in range (min (127, len (self.query_his) - 1), 0, -1):
            if self.query_his [i] == 0:
                break
            query = ["Delete", "read", "insert"] [self.query_his [i] - 1]
            key = self.key_his [i]
            tmp.append ((query, key))
        tmp = tmp [::-1]
        print (tmp)

        print ()

        tmp = []
        for i in range (0, self.nstep):
            tmp.append (["DS_I", "DS_II", "DS_III"] [self.action_his [- i - 1]])
        tmp = tmp [::-1]
        print ("Actions: ", tmp)

        print ()

        tmp = []
        for i in range (min (127, len (self.rewards) - 1), 0, -1):
            tmp.append (self.rewards [i])
        tmp = tmp [::-1]
        print ("Reward:", tmp)

        print ()

        print ("Sum reward: ", self. sum_reward)

        time.sleep (0.5)


    def step (self, action):
        self.nstep += 1

        if self.nstep % 50 == 0:
            self.pool_reset ()
            if self.config ["visualize"]:
                print ("POOL RESETED")

            

        done = False
        if self.nstep >= self.max_step:
            done = True
        reward = 0
        
        self.action_his.append (action)
        if len (self.action_his) > 128:
            self.action_his.pop (0)



        if action == self.next_behavior:
            reward += self.next_cost

        self.sum_reward += reward

        self.rewards.append (reward)
        if len (self.rewards) > 128:
            self.rewards.pop (0)

        self.general_behaviror ();
        # print (self.next_behavior)
        # last_time = self.current_time
        # self.current_time = time.time ()
        # print (self.current_time - last_time)
        if self.config ["visualize"]:
            self.visualize ()  


        info = {}
        ret = (self.observation (), reward, done, info)
        return ret


    def observation (self):
        # obs_action = copy.deepcopy (self.action_his)
        obs = np.concatenate ([np.array (self.key_his, dtype=np.float32) [None] / 100, 
                                np.array (self.query_his, dtype=np.float32) [None] / 3], 0)

        if self.dstype == "test":
            for i in range (10):
                obs [0][i] = 0
                obs [1][i] = 0
        return obs

def test ():
    config = {
        "observation_shape": (10,10,10),
        "max_step": 100,
    }
    env = database_env (config, 2)

    obs = env.reset ()

    print (obs)

    for i in range (10):
        print ("STEP ", i)
        env.debug_pool ()
        action = int (input ("action = "))
        obs, reward, done, info = env.step (action)
        print ("reward = ", reward)
        print (obs)

if __name__ == '__main__':
    test ()