import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

def get_batch(X, size):
    ids = np.random.choice(len(X), size, replace=False)
    return (X[ids], ids)
        
def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

class selfCrossEntropy(nn.Module):
    def __init__(self):
        super(selfCrossEntropy, self).__init__()

    def forward(self, output, target):
        output = nn.functional.softmax(output,dim=0)
        # return -torch.mean(torch.sum(target * torch.log(torch.maximum(output, 1e-10)) \
        #     + (1-target) * torch.log(torch.maximum(1 - output, 1e-10)),1))
        return -torch.mean(torch.sum(target * torch.log(torch.clamp(output, min = 1e-10)) \
            + (1-target) * torch.log(torch.clamp(1 - output, min = 1e-10)),1))

class selfLatentLoss(nn.Module):
    def __init__(self):
        super(selfLatentLoss, self).__init__()

    def forward(self, z_mean, z_log_sigma_sq):
        return  torch.mean(torch.sum(torch.pow(z_mean, 2) + torch.exp(z_log_sigma_sq)- z_log_sigma_sq - 1,1))


class selfVLoss(nn.Module):
    def __init__(self, lambda_v, lambda_r):
        super(selfVLoss, self).__init__()
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
    def forward(self, v, z):
        return  1.0*self.lambda_v/self.lambda_r * torch.mean( torch.sum(torch.pow(v - z, 2), 1))

def get_activaton(activation):
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softmax':
        return nn.Softmax()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'linear':
        raise NotImplementedError("loss {}: not implemented!".format(loss)) 

class MyDataset(Dataset):

    def __init__(self, x, noise=False):
        self.x = x.astype(np.float32)   #array
        self.noise = noise
        self.shape = x.shape

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (self.shape[0], self.shape[1]))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), int(round(
                    frac * len(i))), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def __getitem__(self, index):
        if self.noise:
            return self.x[index], self.add_noise(self.x[index].reshape(1,self.shape[1])).reshape(self.shape[1])
        else:
            return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class CvaeDataset(Dataset):
    def __init__(self, x, z):
        self.x = x.astype(np.float32)   #array
        self.z = z.astype(np.float32)   #array

    def __getitem__(self, index):
            return self.x[index], self.z[index]

    def __len__(self):
        return self.x.shape[0]
