import os
os.getcwd()
os.chdir('/home/dongha0718/sy/odim')


from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset  #loaders
from base.base_net import BaseNet
#from utils.misc import binary_cross_entropy
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
from PIL import Image
from datasets.mnist import Refine_MNIST_Dataset


import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import ConcatDataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from networks.main import build_network


def binarize(inputs):
    inputs[inputs > 0.5] = 1.
    inputs[inputs <= 0.5] = 0.
    return inputs


def odim_light(dataset_name,dataset, filter_net_name, model_seed,seed, logger, train_option, gamma_,qt_,ens,train_n,SSOD=True):

    if 'VAE_alpha1.' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 1.
    elif 'IWAE_alpha1.' in train_option:
        #num_sam = 100
        #num_aggr = 50
        # For CNN
        num_sam = 50
        num_aggr = 10
        alpha = 1.
    elif 'VAE_alpha100.' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 100.
    elif 'IWAE_alpha100.' in train_option:
        num_sam = 100
        num_aggr = 50
        alpha = 100.
    elif 'VAE_alpha0.01' in train_option:
        num_sam = 1
        num_aggr = 1
        alpha = 0.01
    elif 'IWAE_alpha0.01' in train_option:
        num_sam = 100
        num_aggr = 50
        alpha = 0.01
    
    
    import random
    lr_milestone = 50
    weight_decay = 0.5e-6
    optimizer_name = 'adam'
        
    device = 'cuda'

    filter_model_lr = 0.001
    if SSOD:
        tot_filter_model_n_epoch = 100
    
    else:
        tot_filter_model_n_epoch = 90

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.backends.cudnn.deterministic = True
    
    # Initialize DeepSAD model and set neural network phi
    filter_model = build_network(filter_net_name)
    filter_model = filter_model.to(device)
    
    filter_optimizer = optim.Adam(filter_model.parameters(), lr=filter_model_lr, weight_decay=weight_decay)
    # Set learning rate scheduler
    filter_scheduler = optim.lr_scheduler.MultiStepLR(filter_optimizer, milestones=(lr_milestone,), gamma=0.1)

    # Training
    logger.info('Starting train filter_model...')
    start_time = time.time()
    #normal_epoch_loss_list = []
    #abn_epoch_loss_list = []

    #normal_epoch_loss = 0.0
    #abn_epoch_loss = 0.0
    patience_idx = 0
    best_dist = 0
    mb_idx = 0
    n_batches = 0
    start_time = time.time()
    running_time = 0.
    n_jobs_dataloader = 0
    test_auc_sc = []
    train_auc_records = []
    best_loss_matrix = np.zeros((0, 0))
    train_iteration  =5

    m_0 = 128
    gamma = gamma_ 
    qt = qt_
    m_1 = qt * m_0
    epoch_qt = 10
    epoch_means = []
    epoch_weight = []
    for epoch in range(tot_filter_model_n_epoch):
        loss_total = []
        #epoch_loss = 0.0
        #normal_epoch_loss = 0.0
        #abn_epoch_loss = 0.0
        n_batches = 0
        total_target_0 = 0
        total_targets_count = 0
        ratio = 0
        epoch_start_time = time.time()

        if epoch<epoch_qt:
            adaptive_batch_size = m_0
        else:
            if 'CIFAR'in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 3000)
                
            elif 'MNIST-C'in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif 'InternetAds' in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif 'SVHN' in dataset_name:
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 1000)
            elif dataset_name=='mnist':
                if gamma_ > 1.01:
                    adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 5000)
            else:
                adaptive_batch_size = min(int(m_0 * (gamma ** (epoch-epoch_qt))), train_n, 20000)

        if 'all' in dataset_name:
            train_loader = dataset.loaders(batch_size=adaptive_batch_size, num_workers=n_jobs_dataloader)
        else:
            train_loader, test_loader = dataset.loaders(batch_size=adaptive_batch_size, num_workers=n_jobs_dataloader)


        if 'all' in dataset_name:
            train_loader_noshuf = dataset.loaders(batch_size=128, shuffle_train=False, drop_last = False, num_workers=n_jobs_dataloader)
        else:
            train_loader_noshuf, test_loader_noshuf  = dataset.loaders(batch_size=128, shuffle_train=False,num_workers=n_jobs_dataloader)

        if SSOD: 
            filtered_inputs = []
            filtered_targets = []
            filtered_idxs = []


            for inputs, targets, idxs in train_loader:
                mask = (targets == 0)
                filtered_inputs.append(inputs[mask])
                filtered_targets.append(targets[mask])
                filtered_idxs.append(idxs[mask])
                
            filtered_inputs = torch.cat(filtered_inputs)
            filtered_targets = torch.cat(filtered_targets)
            filtered_idxs = torch.cat(filtered_idxs)

            if dataset_name in ['mnist', 'fmnist']:
                filtered_inputs = filtered_inputs.view(filtered_inputs.size(0), -1)

            ssod_inputs = filtered_inputs

            ssod_dataset = torch.utils.data.TensorDataset(ssod_inputs, filtered_targets, filtered_idxs)
            ssod_loader = torch.utils.data.DataLoader(ssod_dataset, batch_size=adaptive_batch_size, shuffle=True)
            
            train_iter = iter(ssod_loader)
        else: 
            train_iter = iter(train_loader)


        train_loss_list = []
        train_targets_list = []
        train_idx_list = []
        ### Train VAE
        for batch_idx in range(train_iteration):
            #break
            try:
                inputs, targets, idx = next(train_iter)

            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets, idx = next(train_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            #break
            if 'binarize' in train_option:
                inputs = binarize(inputs)
            # Zero the network parameter gradients
            filter_model.train()
            filter_optimizer.zero_grad()
            # Update network parameters via backpropagation: forward + backward + optimize
            if 'pixel' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,alpha)
            elif 'gaussian' in filter_net_name:
                loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,alpha)
            elif 'gaussian' in train_option:
                loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,alpha)
            else:
                loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,alpha)

            #####################################################################
            #####################################################################

            train_loss_list.append(loss_vec.data.cpu())  
            train_targets_list.append(targets.cpu().numpy())
            train_idx_list += list(idx.numpy())


            loss_quantile_prob = 1-(np.clip((epoch-epoch_qt),0,1)*(1-m_1/m_0)) 
            loss_quantile = torch.quantile(loss_vec.detach(), loss_quantile_prob)    
            trimming_ind = (loss_vec.detach()<=loss_quantile)   
            trimmed_loss = (loss_vec*trimming_ind).sum()/trimming_ind.sum()   
            filtered_targets = targets[trimming_ind]
            
            trimmed_loss.backward()
            filter_optimizer.step()

  
        if SSOD:
            if epoch > 70:
                patience_idx = 0
                running_time += (time.time() - start_time)

                test_loss_list_eval = []
                test_targets_list_eval = []
                test_idx_list_eval = []

                filter_model.eval()
                for data in test_loader:
                    inputs, targets ,idx = data
                    inputs = inputs.to(device)
                    inputs = inputs.view(inputs.size(0), -1)
                    if 'binarize' in train_option:
                        inputs = binarize(inputs)
                    # Update network parameters via backpropagation: forward + backward + optimize
                    if 'pixel' in filter_net_name:
                        loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,1.)
                    elif 'gaussian' in filter_net_name:
                        loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,1.)
                    elif 'gaussian' in train_option:
                        loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,1.)
                    else:
                        loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,1.)

                    test_loss_list_eval.append(loss_vec.data.cpu())
                    test_targets_list_eval.append(targets.cpu().numpy())
                    test_idx_list_eval += list(idx.numpy())
                   
                        

                test_losses = torch.cat(test_loss_list_eval,0).numpy().reshape(-1,1)
                best_idx = test_idx_list_eval
                Epoch = epoch
                start_time = time.time()
                test_targets_list = np.concatenate(test_targets_list_eval)
                test_loss_list = np.concatenate(test_loss_list_eval)
                auc = roc_auc_score(test_targets_list,test_loss_list)

                test_auc_sc.append(auc)
                minmax_scaler = MinMaxScaler()
                if best_loss_matrix.size == 0 :
                    best_loss_matrix = minmax_scaler.fit_transform(test_losses)
                else:
                    best_loss_matrix = np.hstack((best_loss_matrix, minmax_scaler.fit_transform(test_losses)))


        else:
            if epoch > 70:
                patience_idx = 0
                running_time += (time.time() - start_time)

                train_loss_list_eval = []
                train_targets_list_eval = []
                train_idx_list_eval = []

                filter_model.eval()
                for data in train_loader_noshuf:
                    inputs, targets ,idx = data
                    inputs = inputs.to(device)
                    inputs = inputs.view(inputs.size(0), -1)
                    if 'binarize' in train_option:
                        inputs = binarize(inputs)
                    # Update network parameters via backpropagation: forward + backward + optimize
                    if 'pixel' in filter_net_name:
                        loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var_pixelcnn(inputs , filter_model , num_sam , num_aggr,1.)
                    elif 'gaussian' in filter_net_name:
                        loss,loss_vec = VAE_IWAE_loss_gaussian_mean_var(inputs , filter_model , num_sam , num_aggr,1.)
                    elif 'gaussian' in train_option:
                        loss,loss_vec = VAE_IWAE_loss_gaussian(inputs , filter_model , num_sam , num_aggr,1.)
                    else:
                        loss,loss_vec = VAE_IWAE_loss(inputs , filter_model , num_sam , num_aggr,1.)

                    train_loss_list_eval.append(loss_vec.data.cpu())
                    train_targets_list_eval.append(targets.cpu().numpy())
                    train_idx_list_eval += list(idx.numpy())


                train_losses = torch.cat(train_loss_list_eval,0).numpy().reshape(-1,1)
                best_idx = train_idx_list_eval
                #Epoch = epoch
                start_time = time.time()
                
                minmax_scaler = MinMaxScaler()
                if best_loss_matrix.size == 0 :
                    best_loss_matrix = minmax_scaler.fit_transform(train_losses)
                else:
                    best_loss_matrix = np.hstack((best_loss_matrix, minmax_scaler.fit_transform(train_losses)))


                

        running_time += (time.time() - start_time)
   
        print(epoch)
        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time

    #ens_num = ens 

    mean_loss = np.mean(best_loss_matrix,axis=1)
    mean_loss= torch.tensor(mean_loss)

    '''
    mean_loss = np.mean(best_loss_matrix[:,-ens_num:],axis=1)
    mean_loss= torch.tensor(mean_loss)
    '''

    test_auc_sc = pd.DataFrame({'epoch': epoch, 'auc': test_auc_sc})
    running_time += (time.time() - start_time)
    idx_mean_loss = pd.DataFrame({'idx' : best_idx, 'loss' :mean_loss.numpy()})
    train_auc_records = pd.DataFrame(train_auc_records)
    
    return(idx_mean_loss,running_time)


################################################################################################
def logp_z_normal(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector


################################################################################################
## Calculate log p(z)
################################################################################################
def logp_z(z , z_mu_ps , z_log_var_ps , p_cluster):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : n_pseudo * z_dim
    z_ten = z.unsqueeze(1)                                                              ## mini_batch * 1 * z_dim
    z_mu_ps_ten , z_log_var_ps_ten = z_mu_ps.unsqueeze(0) , z_log_var_ps.unsqueeze(0)   ## 1 * n_pseudo * z_dim
    
    logp = (-z_log_var_ps_ten/2-torch.pow((z_ten-z_mu_ps_ten) , 2)/(2*torch.exp(z_log_var_ps_ten))).sum(2) + \
            (torch.log(p_cluster).unsqueeze(0)) ## logp : mini_batch * n_pseudo
    
    max_logp = torch.max(logp , 1)[0]
    #logp = torch.log((torch.exp(logp)).sum(1))
    
    logp = max_logp + torch.log(torch.sum(torch.exp(logp - max_logp.unsqueeze(1)), 1))
    
    return logp

################################################################################################
## Calculate log p(z|x)
################################################################################################

def logp_z_given_x(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector


################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp

################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z_gaussian(x , x_mu):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    logp = -(1/2)*((x-x_mu)**2).sum(1)
    
    return logp


################################################################################################
## Calculate log p(x|z)
################################################################################################
   
def logp_x_given_z_gaussian_mean_var(x , x_mu, x_logvar):
    ## x : mini_batch * x_dim
    ## x_mu : mini_batch * x_dim

    eps = 1e-5
    #logp = (x * torch.log(x_mu+eps) + (1. - x) * torch.log(1.-x_mu+eps)).sum(1)

    #logp = -(1/2)*((x-x_mu)**2).sum(1)
    logp = (-x_logvar-(1/(2*torch.exp(x_logvar)))*((x-x_mu)**2)).sum(1)
    
    return logp


################################################################################################
## Calculate log pdf of normal distribution
## z ~ N(0,1)
################################################################################################

def logp_z_std_mvn(z):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -torch.pow(z , 2)/2
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate log pdf of normal distribution
################################################################################################

def logq_z(z , z_mu , z_log_var):
    ## z : mini_batch * z_dim
    ## z_mu , z_log_var : mini_batch * z_dim
    
    logp = -z_log_var/2-torch.pow((z-z_mu) , 2)/(2*torch.exp(z_log_var))
    
    return logp.sum(1)  ## (mini_batch,) vector

################################################################################################
## Calculate calculate_center function
################################################################################################
def calculate_center(x , gmvae):   

    z_mu , z_log_var = gmvae.encoder(x)
        
    return z_mu


################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian(x , x_mu_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss

################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var_pixelcnn(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(x,sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss


################################################################################################
## Calculate IWAE loss function
################################################################################################
def VAE_IWAE_loss_gaussian_mean_var(x , gmvae , num_sam , num_aggr,alpha):   

    num_iter = int(num_sam / num_aggr)        
    log_loss_list = []

    x=x.repeat(num_aggr,1)
    
    z_mu , z_log_var = gmvae.encoder(x)

    for h in range(num_iter):
        sam_z , _ = gmvae.encoder.Sample_Z(z_mu , z_log_var)            
        x_mu_sam_z,x_logvar_sam_z = gmvae.decoder(sam_z)
        x_mu_sam_z = x_mu_sam_z.view(x_mu_sam_z.size(0), -1)
        x_logvar_sam_z = x_logvar_sam_z.view(x_logvar_sam_z.size(0), -1)
        ######################################
        ## calculate elbo
        log_p = logp_z_std_mvn(sam_z)
        log_q = logp_z_normal(sam_z , z_mu , z_log_var)
        log_recon = logp_x_given_z_gaussian_mean_var(x , x_mu_sam_z, x_logvar_sam_z)


        elbo = log_recon + alpha*(log_p - log_q)
        log_loss_list.append(elbo.reshape(num_aggr , -1).t())
        

            
    log_loss_list = torch.cat(log_loss_list , 1)
    max_log_loss = torch.max(log_loss_list , 1 , keepdim=True)[0]    
    log_loss_list = torch.exp(log_loss_list - max_log_loss)
    log_loss_list = log_loss_list.mean(1)
    log_loss = torch.log(log_loss_list) + max_log_loss.squeeze(1)
            
    return -log_loss.mean(),-log_loss
