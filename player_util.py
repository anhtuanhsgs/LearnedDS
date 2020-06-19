from __future__ import division
import numpy as np
from utils import normal
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from models.basic_modules import init_linear_lstm

class Agent (object):
    def __init__ (self, model, env, args, state, gpu_id=0):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.gpu_id = gpu_id
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []


    def action_train (self, use_max=False, eps=0.15):
        if self.args.lstm_feats:
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        else:
            value, logit = self.model (Variable(self.state.unsqueeze(0)))


        value = value.unsqueeze (-1).unsqueeze (-1)
        logit = logit.unsqueeze (-1).unsqueeze (-1)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        

        prob_tp = prob.permute (0, 2, 3, 1)
        log_prob_tp = log_prob.permute (0, 2, 3, 1)
        distribution = torch.distributions.Categorical (prob_tp)
        # distribution = torch.distributions.Categorical (torch.clamp (prob_tp, 0.05, 0.95))
        shape = prob_tp.shape
        if not use_max:
            action_tp = distribution.sample ().reshape (1, shape[1], shape[2], 1)
            action = action_tp.permute (0, 3, 1, 2)
            self.action = action.cpu().numpy() [0][0]
            self.actions.append (self.action[0][0])
            log_prob = log_prob.gather(1, Variable(action))
            state, self.reward, self.done, self.info = self.env.step(
                self.action [0][0])
            self.reward = np.array ([[self.reward]], dtype=np.float32)

        if not use_max:
            self.state = torch.from_numpy(state).float()

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward [None][None])
        return self



    def action_test (self):
        with torch.no_grad():
            if self.args.lstm_feats:
                if self.done:
                    self.cx, self.hx = init_linear_lstm (self.args.lstm_feats, self.gpu_id)
                else:
                    self.cx = Variable (self.cx)
                    self.hx = Variable (self.hx)
                value, logit, (self.hx, self.cx) = self.model((Variable (self.state.unsqueeze(0)), (self.hx, self.cx)))
            else:
                value, logit = self.model(Variable (self.state.unsqueeze(0)))
        
        value = value.unsqueeze (-1).unsqueeze (-1)
        logit = logit.unsqueeze (-1).unsqueeze (-1)

        prob = F.softmax (logit, dim=1)

        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0][0][0])
        self.reward = np.array ([[self.reward]], dtype=np.float32)

        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        self.actions.append (action [0])
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        self.mu = []
        self.sigma = []
        return self