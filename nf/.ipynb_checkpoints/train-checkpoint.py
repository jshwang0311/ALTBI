import torch
import logging
import random
import numpy as np
# import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
parent_dir = '/home/jovyan/ALTBI'
sys.path.append(parent_dir)

from datasets import load_dataset
from optim import *


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *


import argparse

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS 
# # # # # # # # # # #

parser = argparse.ArgumentParser(description='ALTBI Experiment')
# arguments for optimization
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--dataset_name_option', type=str, default=None, choices=[None, 'all', 'adbench', 'adbench_all'])
parser.add_argument('--filter_net_name', type=str, default=None)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--ssod', type=bool, default=False)
parser.add_argument('--gamma', type=bool, default=1.03)
parser.add_argument('--qt', type=bool, default=0.92)
parser.add_argument('--ens', type=bool, default=20)
parser.add_argument('--nf', type=str, default='glow', choices=[None, 'planar', 'glow'])
parser.add_argument('--data_root', type=str, default='/home/jovyan/offering-optimizer-datavol-1/data')
parser.add_argument('--total_proc_num', type=int, default=1)
parser.add_argument('--proc_idx', type=int, default=1)
parser.add_argument('--skip', type=int, default=0)

#args = parser.parse_args()
args, unknown = parser.parse_known_args()
## For debugging
# args, unknown = parser.parse_known_args([])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # parameter setting
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    use_cuda = args.use_cuda
    gpu_num = args.gpu_num
    #batch_size = args.batch_size
    ratio_known_normal = 0.0
    ratio_known_outlier = 0.0
    n_known_outlier_classes = 0

    dataset_name_list, data_path_list, train_option_list, filter_net_name_list, \
    ratio_pollution_list, normal_class_list_list, patience_thres_list = gen_hyperparams(args.data_root, 
                                                                                        args.dataset_name,
                                                                                        args.dataset_name_option,
                                                                                        )
    simulation_num = int(len(dataset_name_list)/args.total_proc_num)
    simulation_idx_list = list(range((args.proc_idx-1)*simulation_num,args.proc_idx*simulation_num))
    if args.proc_idx == args.total_proc_num:
        simulation_idx_list = list(range((args.proc_idx-1)*simulation_num, len(dataset_name_list)))

    best_auc = []
    results_summary = []
    train_auc_list_mean = []
    test_auc_list_mean = []
    train_pr_list_mean = []
    test_pr_list_mean = []

    skip_count = 0
    save_metric_dir = f'../Results/nf_{args.nf}'
    os.makedirs(save_metric_dir, exist_ok=True)
    # for dataset_idx in range(len(dataset_name_list)):
    for dataset_idx in simulation_idx_list:
        if skip_count<args.skip:
            skip_count += 1
            csv_file_path = os.path.join(save_metric_dir,f'train_metric_{args.dataset_name_option}_total{args.total_proc_num}_idx{args.proc_idx}.csv')
            try:
                train_df = pd.read_csv(csv_file_path)
                train_df.set_index(keys = 'row_names', inplace = True)
            except:
                print('No Result.')
            continue
        dataset_name = dataset_name_list[dataset_idx]
        data_path = data_path_list[dataset_idx]
        train_option = train_option_list[dataset_idx]
        filter_net_name = filter_net_name_list[dataset_idx]
        ratio_pollution = ratio_pollution_list[dataset_idx]
        normal_class_list = normal_class_list_list[dataset_idx]
        patience_thres = patience_thres_list[dataset_idx]

        data_seed_list = [100, 200, 300, 400, 500]
        start_model_seed = 1234
        n_ens = 1
        normal_class_idx = 0

        for normal_class_idx in range(len(normal_class_list)):
            normal_class = normal_class_list[normal_class_idx]
            known_outlier_class = 0

            results_df = pd.DataFrame()
            ratio = pd.DataFrame()
            train_results = pd.DataFrame()
            # Default device to 'cpu' if cuda is not available
            if not torch.cuda.is_available():
                device = 'cpu'

            torch.cuda.set_device(gpu_num)
            print('Current number of the GPU is %d'%torch.cuda.current_device())

            seed_idx = 0
            nu = 0.1
            num_threads = 0
            n_jobs_dataloader = 0

            row_name_list = []
            for seed_idx in range(len(data_seed_list)):
                row_name = f'Class{normal_class}_simulation{seed_idx+1}'
                row_name_list.append(row_name)
            row_name = f'Average'
            row_name_list.append(row_name)
            row_name = f'Std'
            row_name_list.append(row_name)
            train_auc_list = []
            train_ap_list = []
            test_auc_list = []
            test_ap_list = []
            row_name_list = []

            for seed_idx in range(len(data_seed_list)):
                row_name = f'{dataset_name}_Class{normal_class}_simulation{seed_idx+1}'
                row_name_list.append(row_name)
            row_name = f'{dataset_name}_Average'
            row_name_list.append(row_name)
            row_name = f'{dataset_name}_Std'
            row_name_list.append(row_name)

            for seed_idx in range(len(data_seed_list)):
                seed = data_seed_list[seed_idx]
                save_dir = os.path.join(f'../Results/{dataset_name}/nf_{args.nf}',f'log{seed}')
                os.makedirs(save_dir, exist_ok=True)
                save_score_dir = os.path.join(f'../Results/{dataset_name}/nf_{args.nf}',f'score{seed}')
                os.makedirs(save_score_dir, exist_ok=True)

                # Set up logging
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger()
                logger.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                log_file = save_dir + '/log_'+dataset_name+'_trainOption'+ train_option + '_normal' + str(normal_class) +'.txt'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info('-----------------------------------------------------------------')
                logger.info('-----------------------------------------------------------------')
                # Print paths
                logger.info('Log file is %s' % log_file)
                logger.info('Data path is %s' % data_path)
                # Print experimental setup
                logger.info('Dataset: %s' % dataset_name)
                logger.info('Normal class: %s' % normal_class)
                logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
                logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
                logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
                if n_known_outlier_classes == 1:
                    logger.info('Known anomaly class: %d' % known_outlier_class)
                else:
                    logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
                logger.info('Network: %s' % filter_net_name)


                # Print model configuration
                logger.info('nu-parameter: %.2f' % nu)

                # Set seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                # Load data
                ######################################################################
                ######################################################################
                dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                       ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                       random_state=np.random.RandomState(seed),
                                      train_option=args.nf)###
                ######################################################################
                if 'all' in dataset_name:
                    temp_train_loader = dataset.loaders(batch_size=64, num_workers=n_jobs_dataloader)
                else:
                    temp_train_loader, test_loader = dataset.loaders(batch_size=64, num_workers=n_jobs_dataloader)

                train_ys = []
                train_idxs = []                
                test_ys = []
                test_idxs = []
                for (inputs, outputs, idxs) in temp_train_loader:
                    train_ys.append(outputs.data.numpy())
                    train_idxs.append(idxs.data.numpy())
                input_shape = tuple(inputs.shape)[1:]

                ## not implemented ssod setting
                # if args.ssod == True: 
                #     for (_, outputs, idxs) in test_loader:
                #         test_ys.append(outputs.data.numpy())
                #         test_idxs.append(idxs.data.numpy())
                #     test_ys = np.hstack(test_ys)
                #     test_idxs = np.hstack(test_idxs)
                #     test_idxs_ys = pd.DataFrame({'idx' : test_idxs,'y' : test_ys})

                train_ys = np.hstack(train_ys)
                train_idxs = np.hstack(train_idxs)
                train_idxs_ys = pd.DataFrame({'idx' : train_idxs,'y' : train_ys})
                train_n =  train_ys.shape[0]
                loss_column = ['idx','ens_value','ens_st_value','y']
 
                for model_iter in range(n_ens):
                    model_seed = start_model_seed+(model_iter*5)
                    gamma= args.gamma
                    qt = args.qt
                    ens = args.ens
                    idx_mean_loss,running_time = odim_light(dataset_name,dataset, filter_net_name, 
                                                            model_seed,seed, logger, train_option,
                                                            gamma,qt,ens,train_n,
                                                            nf_option = args.nf, input_shape=input_shape
                                                           )
                    train_me_losses = (idx_mean_loss.to_numpy())[:,1]
                    st_train_me_losses = (train_me_losses - train_me_losses.mean())/train_me_losses.std()
                    idx_mean_loss['st_loss'] = st_train_me_losses
                    add_label_idx_losses = pd.merge(idx_mean_loss, train_idxs_ys, on ='idx')
                    fpr, tpr, thresholds = metrics.roc_curve(np.array(add_label_idx_losses['y']), np.array(add_label_idx_losses['loss']), pos_label=1)
                    roc_auc = metrics.auc(fpr, tpr)

                    if model_iter == 0:
                        ens_loss = add_label_idx_losses
                        ens_loss.columns = loss_column

                    else:
                        merge_data = pd.merge(ens_loss, idx_mean_loss, on = 'idx')
                        merge_data['ens_value'] = merge_data['ens_value'] + merge_data['loss']
                        merge_data['ens_st_value'] = merge_data['ens_st_value'] + merge_data['st_loss']
                        ens_loss = merge_data[loss_column]


                    train_auc = roc_auc_score(np.array(ens_loss['y']), np.array(ens_loss['ens_st_value']))
                    train_ap = average_precision_score(np.array(ens_loss['y']), np.array(ens_loss['ens_value']))

                    # if args.ssod == True:
                    #     test_me_losses = (idx_mean_loss.to_numpy())[:,1]
                    #     st_test_me_losses = (test_me_losses - test_me_losses.mean())/test_me_losses.std()
                    #     idx_mean_loss['st_loss'] = st_test_me_losses
                    #     add_label_idx_test_losses = pd.merge(idx_mean_loss, test_idxs_ys, on ='idx')
                    #     fpr, tpr, thresholds = metrics.roc_curve(np.array(add_label_idx_test_losses['y']), np.array(add_label_idx_test_losses['loss']), pos_label=1)
                    #     roc_auc = metrics.auc(fpr, tpr)
                    #     if model_iter == 0:
                    #         test_ens_loss = add_label_idx_test_losses
                    #         test_ens_loss.columns = loss_column
                    #     else:
                    #         test_merge_data = pd.merge(ens_loss, idx_mean_loss, on = 'idx')
                    #         test_merge_data['ens_value'] = test_merge_data['ens_value'] + test_merge_data['loss']
                    #         test_merge_data['ens_st_value'] = test_merge_data['ens_st_value'] + test_merge_data['st_loss']
                    #         test_ens_loss = test_merge_data[loss_column]
                    #     test_auc = roc_auc_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_st_value']))
                    #     test_ap = average_precision_score(np.array(test_ens_loss['y']), np.array(test_ens_loss['ens_value']))

                # if args.ssod == True: 
                #     test_auc_list.append(test_auc)
                #     test_ap_list.append(test_ap)
                train_auc_list.append(train_auc)
                train_ap_list.append(train_ap)

            train_auc_list.append(np.mean(train_auc_list))
            train_auc_list.append(np.std(train_auc_list))
            train_ap_list.append(np.mean(train_ap_list))
            train_ap_list.append(np.std(train_ap_list))

            class_train_df = pd.DataFrame({
                'row_names' : row_name_list,
                'AUC' : train_auc_list,
                'PRAUC' : train_ap_list
            })
            class_train_df.set_index(keys = 'row_names', inplace = True)
            try:
                train_df = pd.concat([train_df, class_train_df], axis = 0)
            except:
                train_df = class_train_df
            # train_auc_list_mean.append([dataset_name,np.mean(train_auc_list),np.mean(train_ap_list)])
            # train_df= pd.DataFrame(train_auc_list_mean,columns=['Dataset','AUC','PRAUC'])
            # csv_file_path = f'/home/dongha0718/sy/odim/Results2/no_minmax_ens.csv'
            if args.dataset_name_option is None:
                csv_file_path = os.path.join(save_metric_dir,f'train_metric_{dataset_name}.csv')
            else:
                csv_file_path = os.path.join(save_metric_dir,f'train_metric_{args.dataset_name_option}_total{args.total_proc_num}_idx{args.proc_idx}.csv')
            train_df.to_csv(csv_file_path)

#             if args.ssod == True: 
#                 test_auc_list_mean.append([dataset_name,np.mean(test_auc_list),np.mean(test_ap_list)])
#                 test_df= pd.DataFrame(test_auc_list_mean,columns=['Dataset','AUC','PRAUC'])
#             else:
#                 train_auc_list_mean.append([dataset_name,np.mean(train_auc_list),np.mean(train_ap_list)])
#                 train_df= pd.DataFrame(train_auc_list_mean,columns=['Dataset','AUC','PRAUC'])

#             if args.ssod == True: 
#                 csv_file_path = f'/home/dongha0718/sy/odim/Results2/ens_30/ssod/ens_ssod_seed1234.csv'
#                 #test_df.to_csv(csv_file_path)
#             else: 
#                 csv_file_path = f'/home/dongha0718/sy/odim/Results2/no_minmax_ens.csv'
#                 #train_df.to_csv(csv_file_path)



