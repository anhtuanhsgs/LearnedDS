from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import CNN, CNN7
from .conv_lstm import ConvLSTMCell
from .basic_modules import init_linear_lstm
from utils import norm_col_init, weights_init

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x


class ActorCritic (nn.Module):
    def __init__ (self, args, last_feat_ch, backbone, out_ch, lstm_feats=0, gpu_id=0):
        super (ActorCritic, self).__init__ ()
        self.name = backbone.name
        self.backbone = backbone
        if lstm_feats:
            self.lstm = nn.LSTMCell (last_feat_ch, lstm_feats)
            last_feat_ch = lstm_feats
            self.use_lstm = True
        else:
            self.use_lstm = False

        self.critic = nn.Linear (last_feat_ch, 1)
        self.actor = nn.Linear (last_feat_ch, out_ch)

    def forward (self, x):
        if self.use_lstm:
            x, (hx, cx) = x
        x = self.backbone (x)

        if self.use_lstm:
            hx, cx = self.lstm (x, (hx, cx))
            x = hx
        critic = self.critic (x)    
       
        actor = self.actor (x)
        ret = (critic, actor,)

   
        if self.use_lstm:
            ret += ((hx, cx),)
        return ret

class CustomDense (nn.Module):
    def __init__ (self, args, in_ch, features):
        super (CustomDense, self).__init__ ()
        self.name = "CustomDense"
        self.history = nn.Sequential (
                nn.Conv1d (in_channels=in_ch, out_channels=features [0], kernel_size=11, stride=5), 
                nn.Conv1d (in_channels=features [0], out_channels=features [1], kernel_size=7, stride=3),
                nn.Conv1d (in_channels=features [1], out_channels=features [2], kernel_size=5, stride=2),
            )
        self.final = nn.Linear (10, 32)        


    def forward (self, x):
        next_query = x [:,:,-1]
        ret = self.history (x)
        ret = ret.view (ret.size (0), -1)
        next_query = next_query.view (next_query.size (0), -1)
        ret = torch.cat ([next_query, ret], 1)
        ret = self.final (ret)

        return ret

def get_model (args, name, input_shape, features, num_actions, lstm_feats=0, gpu_id=0):
    backbone = CustomDense (args, 2, features=[32, 16, 8])
    model = ActorCritic (args, 32, backbone, num_actions, lstm_feats, gpu_id)
    return model


def test (args):
    x = torch.randn ((1,2,128), dtype=torch.float32)
    args = {}
    model = get_model (args, "tmp", x.shape, [32, 16, 8], num_actions=3)
    # model = CustomDense (args, 2, [32,16,8])
    y = model (x)
    # print (y.shape)
    print (y [0].shape, y [1].shape)

