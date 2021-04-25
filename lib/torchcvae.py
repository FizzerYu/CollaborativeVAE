import numpy as np
from lib.utils import noise_validator, RMSELoss, selfCrossEntropy
from lib.utils import selfLatentLoss, get_activaton, selfVLoss, MyDataset, CvaeDataset
# import tensorflow as tf
import sys
import math
import scipy
import scipy.io
import logging
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader

class Params:
    """Parameters for DMF
    """
    def __init__(self):
        self.a = 1
        self.b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 10
        self.lambda_r = 1
        self.max_iter = 10
        self.M = 300

        # for updating W and b
        self.lr = 0.001
        self.batch_size = 128
        self.n_epochs = 10

class inference_generation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_z, activation):
        super(inference_generation, self).__init__()
        # rec
        activation = [get_activaton(x) for x in activation]
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim[0]), activation[0],
                                     nn.Linear( hidden_dim[0], hidden_dim[1]), activation[1])
        self.fc_z_mean = nn.Linear(hidden_dim[1], n_z)
        self.fc_z_log_sigma = nn.Linear(hidden_dim[1], n_z)
        #gen
        self.decoder = nn.Sequential(nn.Linear(n_z, hidden_dim[1]), activation[0],
                                     nn.Linear( hidden_dim[1], hidden_dim[0]), activation[1])
        self.fc_gen = nn.Linear(hidden_dim[0], in_dim)
        # self.weights_init(init_weight, init_de_weight)

    def forward(self, x):  #[b, in_dim]
        x = self.encoder(x)
        z_mean = self.fc_z_mean(x)                  # mu
        z_log_sigma_sq = self.fc_z_log_sigma(x)     # log_var
        z = self.reparameterize(z_mean, z_log_sigma_sq)
        x_recon = self.decoder(z)
        x_recon = self.fc_gen(x_recon)
        x_recon = nn.functional.softmax(x_recon, dim=0)
        return x_recon, z_mean, z_log_sigma_sq, z

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.clamp(torch.exp(log_var), min = 1e-10))
        eps = torch.randn_like(std)
        return mu + eps * std

    # def weights_init(self, init_weight,init_de_weight):
    #     a=0

    def transform(self, x):
        x = self.encoder(x)
        z_mean = self.fc_z_mean(x)                  # mu
        return z_mean  



class CVAE:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, 
        dims, activations, n_z=50, loss_type='cross-entropy', lr=0.1, 
        wd=1e-4, dropout=0.1, random_seed=0, print_step=50, verbose=True):
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors

        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.m_theta = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)

        self.input_dim = input_dim
        self.dims = dims
        self.activations = activations
        self.lr = lr
        self.params = params
        self.print_step = print_step
        self.verbose = verbose
        self.loss_type = loss_type
        self.n_z = n_z
        self.weights = []
        self.reg_loss = 0

    #     self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
    #     self.v = tf.placeholder(tf.float32, [None, self.m_num_factors])

        self.model = inference_generation(self.input_dim, self.dims, self.n_z, self.activations)  # 构建模型
        self.model = self.model.cuda()
        # loss
        # reconstruction loss
        if loss_type == 'rmse':
            self.gen_loss = RMSELoss()
        elif loss_type == 'cross-entropy':
            self.gen_loss = selfCrossEntropy()

        self.latent_loss = selfLatentLoss()
        self.v_loss = selfVLoss(params.lambda_v, params.lambda_r)

        # self.loss = self.gen_loss + self.latent_loss + self.v_loss + 2e-4*self.reg_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Initializing the tensor flow variables
        # self.saver = tf.train.Saver(self.weights)
        # init = tf.global_variables_initializer()

    #     # Launch the session
    #     self.sess = tf.Session()
    #     self.sess.run(init)

    def load_model(self, weight_path, pmf_path=None):
        logging.info("Loading weights from " + weight_path)
        self.model.load_state_dict(torch.load(weight_path))
        if pmf_path is not None:
            logging.info("Loading pmf data from " + pmf_path)
            data = scipy.io.loadmat(pmf_path)
            self.m_U[:] = data["m_U"]
            self.m_V[:] = data["m_V"]
            self.m_theta[:] = data["m_theta"]



    def cdl_estimate(self, data_x, num_iter):
        dataloader = DataLoader(CvaeDataset(data_x, self.m_V), batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for iter, (data_x, data_v) in enumerate(dataloader):  # 一个bs
            data_x = Variable(data_x).cuda()
            data_v = Variable(data_v).cuda()
            x_recon, z_mean, z_log_sigma_sq, z = self.model(data_x)
            # loss = gen_loss(x_recon, data_x) + latent_loss(z_mean, z_log_sigma_sq) + v_loss(data_v, z) + 2e-4*reg_loss()   # reg_loss就是normalization，未实现
            genloss = self.gen_loss(x_recon, data_x)
            vloss = self.v_loss(data_v, z) 
            loss = genloss + vloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            # print("Iter:", '%04d' % (iter+1), \
            #         "loss=", "{:.5f}".format(loss.item()), \
            #         "genloss=", "{:.5f}".format(genloss.item()), \
            #         "vloss=", "{:.5f}".format(vloss.item()))
        return genloss.item()


    def transform(self, data_x):
        with torch.no_grad():
            data_x = torch.from_numpy(data_x.astype(np.float32))
            data_x = Variable(data_x).cuda()
            data_en = self.model.transform(data_x)
        return data_en.cpu().numpy()

    def run(self, users, items, test_users, test_items, data_x, params):
        self.m_theta[:] = self.transform(data_x)                # 获取均值 and 降维
        self.m_V[:] = self.m_theta
        n = data_x.shape[0]
        for epoch in range(params.n_epochs):
            num_iter = int(n / params.batch_size)
            gen_loss = self.cdl_estimate(data_x, num_iter)
            self.m_theta[:] = self.transform(data_x)  # 获取均值 and 降维
            likelihood = self.pmf_estimate(users, items, test_users, test_items, params)
            loss = -likelihood + 0.5 * gen_loss * n * params.lambda_r
            logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
                epoch, loss, -likelihood, gen_loss))

    def pmf_estimate(self, users, items, test_users, test_items, params):
        """
        users: list of list
        """
        min_iter = 1
        a_minus_b = params.a - params.b
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)
        it = 0
        while ((it < params.max_iter and converge > 1e-6) or it < min_iter):
            likelihood_old = likelihood
            likelihood = 0
            # update U
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0
            v = self.m_V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * params.b + np.eye(self.m_num_factors) * params.lambda_u

            for i in range(self.m_num_users):
                item_ids = users[i]
                n = len(item_ids)
                if n > 0:
                    A = np.copy(XX)
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids,:])*a_minus_b
                    x = params.a * np.sum(self.m_V[item_ids, :], axis=0)
                    self.m_U[i, :] = scipy.linalg.solve(A, x)
                    
                    likelihood += -0.5 * params.lambda_u * np.sum(self.m_U[i]*self.m_U[i])

            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.m_U[ids]
            XX = np.dot(u.T, u) * params.b
            for j in range(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m>0 :
                    A = np.copy(XX)
                    A += np.dot(self.m_U[user_ids,:].T, self.m_U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.a * np.sum(self.m_U[user_ids, :], axis=0) + params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)
                    
                    likelihood += -0.5 * m * params.a
                    likelihood += params.a * np.sum(np.dot(self.m_U[user_ids, :], self.m_V[j,:][:, np.newaxis]),axis=0)
                    likelihood += -0.5 * self.m_V[j,:].dot(B).dot(self.m_V[j,:][:,np.newaxis])

                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep) 
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.m_num_factors) * params.lambda_v
                    x = params.lambda_v * self.m_theta[j,:]
                    self.m_V[j, :] = scipy.linalg.solve(A, x)
                    
                    ep = self.m_V[j,:] - self.m_theta[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
            
            it += 1
            converge = abs(1.0*(likelihood - likelihood_old)/likelihood_old)

            if self.verbose:
                if likelihood < likelihood_old:
                    print("likelihood is decreasing!")

                print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood

    # def activate(self, linear, name):
    #     if name == 'sigmoid':
    #         return tf.nn.sigmoid(linear, name='encoded')
    #     elif name == 'softmax':
    #         return tf.nn.softmax(linear, name='encoded')
    #     elif name == 'linear':
    #         return linear
    #     elif name == 'tanh':
    #         return tf.nn.tanh(linear, name='encoded')
    #     elif name == 'relu':
    #         return tf.nn.relu(linear, name='encoded')

    # def save_model(self, weight_path, pmf_path=None):
    #     self.saver.save(self.sess, weight_path)
    #     logging.info("Weights saved at " + weight_path)
    #     if pmf_path is not None:
    #         scipy.io.savemat(pmf_path,{"m_U": self.m_U, "m_V": self.m_V, "m_theta": self.m_theta})
    #         logging.info("Weights saved at " + pmf_path)


