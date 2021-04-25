import numpy as np
# import tensorflow as tf
import scipy.io
import logging
import argparse
import os 
import torch 
from lib.torchvae import VariationalAutoEncoder
# from lib.vae import VariationalAutoEncoder
# from lib.utils import *

def set_logger(args):
    log_file = os.path.join(args.save_path, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True


def ArgParser():
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    # parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    # parser.add_argument('-m', '--maxiter', type=int, default=5, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', type=int, default=-1, help='select gpu id, -1 is not using')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # parser.add_argument('--log', type=int, default=1, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--dir', help='dataset directory', default='/home/lichangyv/code/recommendSystem/data')
    # parser.add_argument('--data', help='specify dataset', default='music')
    # parser.add_argument('--layer', nargs='+', help='number of neurals in each layer', type=int, default=[100,20])
    # parser.add_argument('-N', help='number of recommended items', type=int, default=20)
    # parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    # parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=10)
    # parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=0.1)
    # parser.add_argument('--rating', help='feed input as rating', action='store_true')
    # parser.add_argument('--save', help='save model', action='store_false')
    parser.add_argument('--save_path', help='save path', default='/home/lichangyv/code/cvae/torch/experiment')
    # parser.add_argument('--load', help='load model', type=int, default=0)
    parser.add_argument('--print_on_screen', help='print_on_screen', type=bool, default=True)
    parser.add_argument('--model_summary', help='model_summary', type=bool, default=True)
    return parser


if __name__ == "__main__":

	args = ArgParser().parse_args()
	set_logger(args)
	logging.info(args)
	set_global_seed(args.seed)

	os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	args.device = torch.device("cuda" if args.gpu!=-1 and torch.cuda.is_available() else "cpu")
	variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
	data = variables['X']
	# print(data.shape)
	# print(data)
	idx = np.random.rand(data.shape[0]) < 0.8
	train_X = data[idx]
	# print(np.unique(data))
	test_X = data[~idx]
	logging.info('initializing sdae model')
	model = VariationalAutoEncoder(input_dim=8000, dims=[200, 100], z_dim=50, 
		activations=['sigmoid','sigmoid'], epoch=[50, 50], 
		noise='mask-0.3' ,loss='rmse', lr=0.01, batch_size=128, print_step=1)  # loss 原来是cross-entropy 改成了 rmse
	logging.info('fitting data starts...')
	model.fit(train_X, test_X)




# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)
