from __future__ import division

from scipy.ndimage import distance_transform_edt
from setproctitle import setproctitle as ptitle
import torch
from environment import database_env
from utils import setup_logger
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from Utils.Logger import Logger
from Utils.img_aug_func import color_generator
from Utils.utils import *
from utils import adjusted_rand_index
from utils import ScalaTracker
import numpy as np
from models.models import get_model
import torch.nn.functional as F


def test (args, shared_model, env_conf):
    ptitle ('Valid agent')

    if args.valid_gpu < 0:
        gpu_id = args.gpu_ids [-1]
    else:
        gpu_id = args.valid_gpu
    env_conf ["env_gpu"] = gpu_id

    log = {}
    logger = Logger (args.log_dir)

    create_dir (args.log_dir + "models/")

    os.system ("cp *.sh " + args.log_dir)
    os.system ("cp *.py " + args.log_dir)
    os.system ("cp models/models.py " + args.log_dir + "models/")
    os.system ("cp models/basic_modules.py " + args.log_dir + "models/")

    setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars (args)
    env_conf_log = env_conf

    for k in d_args.keys ():
        log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))
    for k in env_conf_log.keys ():
        log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, env_conf_log[k]))


    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)
    
    env = database_env (env_conf, seed=0, dstype="test")
    env.max_step = 900

    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None, gpu_id)
    player.gpu_id = gpu_id

    player.model = get_model (args, args.model, env_conf ["observation_shape"], 
                        args.features, env_conf ["num_actions"], gpu_id=0, lstm_feats=args.lstm_feats)

    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()

    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.state = player.state.cuda()
    player.model.eval ()

    flag = True
    create_dir (args.save_model_dir)


    recent_episode_scores = ScalaTracker (100)
    max_score = 0

    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            player.model.eval ()
            flag = False

        player.action_test ()

        reward_sum += player.reward.mean ()

        if player.done:
            flag = True
            num_tests += 1

            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests

            log ['{}_log'.format (args.env)].info (
                "VALID: Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
                format (
                    time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                    reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores.push (reward_sum)

            if args.save_max and recent_episode_scores.mean () >= max_score:
                max_score = recent_episode_scores.mean ()
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = {}
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

            if num_tests % args.log_period == 0:
                print ("------------------------------------------------")
                print (args.env)
                print ("Log test #:", num_tests)
                print ("sum rewards: ", player.env.sum_reward)
                print ("action_history\n", player.env.action_his)
                print ()
                print ("------------------------------------------------")

                log_info = {
                        'mean_reward': reward_mean,
                        '100_mean_reward': recent_episode_scores.mean ()}
                for tag, value in log_info.items ():
                    logger.scalar_summary (tag, value, num_tests)

            reward_sum = 0
            player.eps_len = 0
            
            player.clear_actions ()
            state = player.env.reset ()

            time.sleep (15)

            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()









        

