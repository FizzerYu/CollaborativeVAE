import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


class run_all(nn.Module):  # AE
    def __init__(self, in_dim, hidden_dim, z_dim, activation):
        super(run_all, self).__init__()
        # rec
        activation = activation
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim[0]), activation[0],
                                     nn.Linear( hidden_dim[0], hidden_dim[1]), activation[1])
        self.fc_z_mean = nn.Linear(hidden_dim[1], z_dim)
        self.fc_z_log_sigma = nn.Linear(hidden_dim[1], z_dim)
        #gen
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden_dim[1]), activation[0],
                                     nn.Linear( hidden_dim[1], hidden_dim[0]), activation[1])
        self.fc_gen = nn.Linear(hidden_dim[0], in_dim)
        # self.weights_init()

    def forward(self, x):  #[b, in_dim]
        x = self.encoder(x)
        z_mean = self.fc_z_mean(x)                  # mu
        z_log_sigma_sq = self.fc_z_log_sigma(x)     # log_var
        z = self.reparameterize(z_mean, z_log_sigma_sq)
        x_recon = self.decoder(z)
        x_recon = self.fc_gen(x_recon)
        x_recon = nn.functional.softmax(x_recon)
        return x_recon, z_mean, z_log_sigma_sq

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.clamp(torch.exp(log_var), min = 1e-10))
        eps = torch.randn_like(std)
        return mu + eps * std
    def weights_init():
        self.encoder[0] = torch.nn.Parameter()
        self.encoder[2] = torch.nn.Parameter()

        self.encoder[0] = torch.nn.Parameter()
        self.encoder[2] = torch.nn.Parameter()






model = run_all(10, [100,200],20,[nn.ReLU(), nn.ReLU()])
print(model.encoder[2])
