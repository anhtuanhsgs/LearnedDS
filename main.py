from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp

from environment import database_env
import os, sys, glob, time
import argparse
from utils import get_data
from Utils.img_aug_func import *
from Utils.utils import *
from models.models import get_model
from shared_optim import SharedRMSprop, SharedAdam
import skimage.io as io

from train import train
from test import test
import copy


parser = argparse.ArgumentParser(description='A3C')

parser.add_argument(
	'--env',
	default='merge_err',
	metavar='ENV',
)

parser.add_argument(
	'--lr',
	type=float,
	default=0.0001,
	metavar='LR',
	help='learning rate (default: 0.0001)')

parser.add_argument(
	'--gamma',
	type=float,
	default=0.98,
	metavar='G',
	help='discount factor for rewards (default: 1)')

parser.add_argument(
	'--tau',
	type=float,
	default=1.00,
	metavar='T',
	help='parameter for GAE (default: 1.00)')

parser.add_argument (
	'--seed',
	type=int,
	default=5
)

parser.add_argument (
	'--max-step',
	type=int,
	default=300
)

parser.add_argument (
	'--DEBUG',
	action="store_true"
)


parser.add_argument (
	'--SEMI-DEBUG',
	action="store_true"
)


parser.add_argument (
	'--features',
	type=int,
	default=[128, 64],
	nargs='+'
)

parser.add_argument (
	'--lstm-feats',
	type=int,
	default=0,
	metavar='HF'
)

parser.add_argument (
	'--num-actions',
	type=int,
	default=3
)

parser.add_argument(
	'--optimizer',
	default='Adam',
	metavar='OPT',)


parser.add_argument(
	'--shared-optimizer',
	action='store_true'
)

parser.add_argument(
	'--amsgrad',
	default=True,
	metavar='AM',
	help='Adam optimizer amsgrad parameter')

parser.add_argument (
	'--model',
	default='Dense',
	choices=["Dense",]
)

parser.add_argument(
	'--log-period',
	type=int,
	default=10,
	metavar='LP',
	help='Log period')

parser.add_argument(
	'--train-log-period',
	type=int,
	default=50,
	metavar='LP',
	help='Log period')

parser.add_argument(
	'--save-period',
	type=int,
	default=100000,
	metavar='SP',
	help='Save period')

parser.add_argument(
	'--gpu-ids',
	type=int,
	default=-1,
	nargs='+',
	help='GPUs to use [-1 CPU only] (default: -1)')


parser.add_argument(
	'--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
	'--save-model-dir',
	default='trained_models/',
	metavar='SMD',
	help='folder to save trained models')


parser.add_argument(
	'--load', default=False, metavar='L', help='load a trained model')


parser.add_argument(
	'--workers',
	type=int,
	default=4,
	metavar='W',
	help='how many training processes to use (default: 32)')

parser.add_argument (
	'--valid-gpu',
	type=int,
	default=-1
)

parser.add_argument(
	'--num-steps',
	type=int,
	default=6,
	metavar='NS',
	help='number of forward steps in A3C (default: 20)')

parser.add_argument (
	'--save-max',
	action='store_true'
)

parser.add_argument (
	'--visualize',
	action='store_true'
)


parser.add_argument (
	'--entropy-alpha',
	type=float,
	default=0.05
)


def setup_env_conf (args):

	env_conf = {
		"max_step": args.max_step,
		"num_actions": args.num_actions,
		"DEBUG": args.DEBUG,
		"visualize": args.visualize,
	}

	env_conf ["observation_shape"] = [128, 2]

	args.env = "DS"

	args.log_dir += args.env + "/"
	args.save_model_dir += args.env + "/"
	create_dir (args.save_model_dir)
	create_dir (args.log_dir)

	return env_conf

if __name__ == '__main__':
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	if args.gpu_ids == -1:
		args.gpu_ids = [-1]
	else:
		torch.cuda.manual_seed(args.seed)
		mp.set_start_method('spawn')

	env_conf = setup_env_conf (args)

	shared_model = get_model (args, args.model, env_conf ["observation_shape"],
							  args.features, env_conf["num_actions"], gpu_id=-1, lstm_feats=args.lstm_feats)

	if args.load:
		saved_state = torch.load(
			args.load,
			map_location=lambda storage, loc: storage)
		shared_model.load_state_dict(saved_state)
	shared_model.share_memory()

	if args.shared_optimizer:
		if args.optimizer == 'RMSprop':
			optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
		if args.optimizer == 'Adam':
			optimizer = SharedAdam(
				shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
		optimizer.share_memory()
	else:
		optimizer = None

	processes = []
	
	p = mp.Process(target=test, args=(args, shared_model, env_conf))
	p.start()
	processes.append(p)
	time.sleep(1)

	for rank in range(0, args.workers):
		p = mp.Process(target=train, args=(rank, args, shared_model, optimizer, env_conf))
		p.start()
		processes.append(p)
		time.sleep(1)
	for p in processes:
		time.sleep(0.1)
		p.join()

