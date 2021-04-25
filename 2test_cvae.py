import numpy as np
# import tensorflow as tf
import scipy.io
import logging
import argparse
import os 
import torch 
from lib.torchcvae import CVAE,Params
def set_logger(args):
    log_file = os.path.join(args.save_path, 'cave.log')
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
    parser.add_argument('--gpu', type=int, default=-1, help='select gpu id, -1 is not using')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_path', help='save path', default='/home/lichangyv/code/cvae/torch/experiment')
    parser.add_argument('--print_on_screen', help='print_on_screen', type=bool, default=True)
    parser.add_argument('--model_summary', help='model_summary', type=bool, default=True)
    return parser

def load_cvae_data():
    data = {}
    data_dir = "data/citeulike-a/"
    variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
    data["content"] = variables['X']
    data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
    data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
    data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")
    data["test_items"] = load_rating(data_dir + "cf-test-1-items.dat")
    return data

def load_rating(path):
  arr = []
  for line in open(path):
      a = line.strip().split()
      if a[0]==0:
          l = []
      else:
          l = [int(x) for x in a[1:]]
      arr.append(l)
  return arr

if __name__ == "__main__":

    args = ArgParser().parse_args()
    set_logger(args)
    logging.info(args)
    set_global_seed(args.seed)

    params = Params()
    params.lambda_u = 0.1
    params.lambda_v = 10
    params.lambda_r = 1
    params.a = 1
    params.b = 0.01
    params.M = 300
    params.n_epochs = 100
    params.max_iter = 1

    data = load_cvae_data()
    num_factors = 50
    model = CVAE(num_users=5551, num_items=16980, num_factors=num_factors, params=params,\
        input_dim=8000, dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'],\
        loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
    # model.load_model(weight_path="model/pretrain")

    model.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"],
    data["content"], params)
    # model.save_model(weight_path="model/cvae", pmf_path="model/pmf")
