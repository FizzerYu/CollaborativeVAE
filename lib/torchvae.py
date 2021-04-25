import torch
import torch.nn as nn
import numpy as np
from lib.utils import noise_validator, RMSELoss, selfCrossEntropy, selfLatentLoss, get_activaton, MyDataset
import logging
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchinfo import summary


allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']


class run_model(nn.Module):  # AE
    def __init__(self, in_dim, hidden_dim, activation):
        super(run_model, self).__init__()
        activation = get_activaton(activation)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.Softmax())
        # init
        self.weights_init(self.encoder)
        self.weights_init(self.decoder)

    def forward(self, x):  #[b, in_dim]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def infer(self, x):
        return self.encoder(x)

    def weights_init(self,block):
        for m in block.children(): # 这里初始化有点问题，原文decoder的初始化为 'weights': tf.transpose(encode['weights'])
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
         
class run_latent(nn.Module):  # AE
    def __init__(self, in_dim, hidden_dim):
        super(run_latent, self).__init__()
        self.fc_z_mean = nn.Linear(in_dim, hidden_dim)
        self.fc_z_log_sigma = nn.Linear(in_dim, hidden_dim)
        self.fc_gen = nn.Linear(hidden_dim, in_dim)
        self.weights_init()

    def forward(self, x):  #[b, in_dim]
        z_mean = self.fc_z_mean(x)                  # mu
        z_log_sigma_sq = self.fc_z_log_sigma(x)     # log_var
        z = self.reparameterize(z_mean, z_log_sigma_sq)
        x_recon = self.fc_gen(z)
        x_recon = nn.functional.softmax(x_recon, dim=0)
        return x_recon, z_mean, z_log_sigma_sq

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.clamp(torch.exp(log_var), min = 1e-10))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def weights_init(self):
        nn.init.xavier_uniform_(self.fc_z_mean.weight)
        nn.init.constant_(self.fc_z_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_z_log_sigma.weight)
        nn.init.constant_(self.fc_z_log_sigma.bias, 0)
        nn.init.xavier_uniform_(self.fc_gen.weight)
        nn.init.constant_(self.fc_gen.bias, 0)

class run_all(nn.Module):  # AE
    def __init__(self, in_dim, hidden_dim, z_dim, activation, init_weight, init_de_weight):
        super(run_all, self).__init__()
        # rec
        activation = [get_activaton(x) for x in activation]
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim[0]), activation[0],
                                     nn.Linear( hidden_dim[0], hidden_dim[1]), activation[1])
        self.fc_z_mean = nn.Linear(hidden_dim[1], z_dim)
        self.fc_z_log_sigma = nn.Linear(hidden_dim[1], z_dim)
        #gen
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden_dim[1]), activation[0],
                                     nn.Linear( hidden_dim[1], hidden_dim[0]), activation[1])
        self.fc_gen = nn.Linear(hidden_dim[0], in_dim)
        self.weights_init(init_weight, init_de_weight)

    def forward(self, x):  #[b, in_dim]
        x = self.encoder(x)
        z_mean = self.fc_z_mean(x)                  # mu
        z_log_sigma_sq = self.fc_z_log_sigma(x)     # log_var
        z = self.reparameterize(z_mean, z_log_sigma_sq)
        x_recon = self.decoder(z)
        x_recon = self.fc_gen(x_recon)
        x_recon = nn.functional.softmax(x_recon, dim=0)
        return x_recon, z_mean, z_log_sigma_sq

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.clamp(torch.exp(log_var), min = 1e-10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def weights_init(self, init_weight,init_de_weight):
        self.encoder[0].load_state_dict(init_weight[0])
        self.encoder[2].load_state_dict(init_weight[1])
        self.fc_z_mean.load_state_dict(init_weight[2])
        self.fc_z_log_sigma.load_state_dict(init_weight[3])
        self.decoder[0].load_state_dict(init_de_weight[2])

        self.decoder[2].weight = torch.nn.Parameter(init_weight[1]['weight'].transpose(1,0))
        self.decoder[2].bias = torch.nn.Parameter(init_weight[0]['bias'])

        self.fc_gen.weight = torch.nn.Parameter(init_weight[0]['weight'].transpose(1,0))
        nn.init.constant_(self.fc_gen.bias, 0)


def getloss(loss):
    if loss == 'cross-entropy':
        return nn.CrossEntropyLoss()
        # return selfCrossEntropy()
    elif loss == 'rmse':
        return RMSELoss()
    else:
        raise NotImplementedError("loss {}: not implemented!".format(loss)) 






class VariationalAutoEncoder:
    """A deep variational autoencoder"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, input_dim, dims, z_dim, activations, epoch=1000, noise=None, loss='cross-entropy',
                 lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = [2, 2]  #epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)   #len(dims)
        self.n_z = z_dim
        self.input_dim = input_dim
        self.weights = []
        self.de_weights = []


    def fit(self, data_x, x_valid):
        
        # valid_dataloader = DataLoader(MyDataset(x_valid, self.noise), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)

        # run
        x = data_x
        for i in range(self.depth):     # 运行两次，获得两个weight
            print('==========>',i, x.shape)
            train_dataloader = DataLoader(MyDataset(x, self.noise), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
            model = run_model(in_dim=x.shape[1], activation=self.activations[i], hidden_dim=self.dims[i])
            model = model.cuda()
            # summary(model, input_size=(12,x.shape[1]),col_names=["input_size", "kernel_size", "output_size"])
            criterion = getloss(self.loss)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            logging.info('Layer {0}'.format(i + 1))
            for now_epoch in range(self.epoch[i]):
                for iter, (orig, noise) in enumerate(train_dataloader):
                    orig = Variable(orig).cuda()
                    noise = Variable(noise).cuda()
                    output = model(noise)
                    loss = criterion(orig, output)
                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()     
                logging.info('==> run_model epoch:{} loss:{:.4f}'.format(now_epoch, loss.item()))
            with torch.no_grad():
                x = model.infer(torch.from_numpy(x.astype(np.float32)).cuda())  # 原始数据
                x = x.cpu().numpy()
            self.weights.append(model.encoder[0].state_dict())
            self.de_weights.append(model.decoder[0].state_dict())

        print(data_x.shape, x.shape, x.dtype)        # (13595, 8000) torch.Size([13595, 200]) torch.float32
        
        # # fit latent layer
        # run_latent
        train_dataloader = DataLoader(MyDataset(x, noise = False), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        model = run_latent(in_dim = x.shape[1], hidden_dim = self.n_z)
        model = model.cuda()
        # summary(model, input_size=(12,x.shape[1]),col_names=["input_size", "kernel_size", "output_size"])
        gen_criterion = selfCrossEntropy()
        latent_criterion = selfLatentLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for i in range(2):  # 50
            for iter, data in enumerate(train_dataloader):
                data = Variable(data).cuda()
                output, z_mean, z_log_sigma_sq = model(data)
                gen_loss = gen_criterion(output, data)
                latent_loss = latent_criterion(z_mean, z_log_sigma_sq)
                loss = gen_loss + 0.5 * latent_loss
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
            logging.info('==> run_latent epoch:{} loss:{:.4f}'.format(i, loss.item()))
        self.weights.append(model.fc_z_mean.state_dict())
        self.weights.append(model.fc_z_log_sigma.state_dict())
        self.de_weights.append(model.fc_gen.state_dict())

        # run_all
        train_dataloader = DataLoader(MyDataset(data_x, noise = False), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        valid_dataloader = DataLoader(MyDataset(x_valid, noise = False), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        model = run_all(in_dim = data_x.shape[1], hidden_dim = self.dims, z_dim = self.n_z,  activation=self.activations,
                        init_weight = self.weights, init_de_weight = self.de_weights)
        model = model.cuda()
        # summary(model, input_size=(12,x.shape[1]),col_names=["input_size", "kernel_size", "output_size"])
        gen_criterion = selfCrossEntropy()
        latent_criterion = selfLatentLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for i in range(2):  # 100
            for iter, data in enumerate(train_dataloader):
                data = Variable(data).cuda()
                output, z_mean, z_log_sigma_sq = model(data)
                gen_loss = gen_criterion(output, data)
                latent_loss = latent_criterion(z_mean, z_log_sigma_sq)
                loss = gen_loss + 0.5 * latent_loss
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
            logging.info('==> run_all epoch:{} loss:{:.4f}'.format(i, loss.item()))
            # valid
        
        # save model

