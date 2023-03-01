#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import random
import time
import pickle
import numpy as np
from tqdm import tqdm
import re
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP
from utils import get_dataset, average_weights, exp_details, powersettool, shapley, \
                  calculate_gradients, update_weights_from_gradients


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time = time.time()  # start the timer
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # BUILD MODEL
    # if args.model == 'cnn':
    #     # Convolutional neural network
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)

    if args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[1][0][0].shape  
        
        # print(img_size) # print check 

        len_in = 1
        for x in img_size:
            len_in *= x
        # print('number of dimension_in is :', len_in) # print check
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)    
    # else:
    #     exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)



    ######## Timing starts ########
    start_time = time.time() # start the timer

    # Powerset list 
    powerset = list(powersettool(range(1,6))) #generate a powerset list of tuples eg [(),(1,),(2),(1,2)]

    # Initialize all the sub-models
    submodel_dict = {}

    for subset in powerset[:-1]: #exclude only the global set. the null set still has a random initialized model
        submodel_dict[subset] = copy.deepcopy(global_model)
        submodel_dict[subset].to(device)
        submodel_dict[subset].train()


    # dictionary containing all the accuracies for the subsets    
    accuracy_dict = {}
    # accuracy_dict[()] = 0

    totalRunTime = time.time() - start_time
    ######## Timing ends ########
    '''
    # get users' history Loss value and select_times
    user_number= len(user_groups)
    user_index = []
    Cost=[]
    for index in range(len(user_groups)):
        user_index.append(index + 1)
        Cost.append(len(user_groups[index+1])/1000)

    history_loss = []
    select_times = []
    q_j = []
    with open('../save/MRfed_OurMNIST_mlp_5_4_old.txt', 'r') as f:
        for line in f.readlines():
            if re.match('Loss *', line) != None:
                line = line.strip('\n')
                line = line.replace('Loss value: ', '')
                history_loss.append(line)
            elif re.match('Select *', line) != None:
                line = line.strip('\n')
                line = line.replace('Select times: ', '')
                select_times.append(line)
            elif re.match('q_j *', line) != None:
                line = line.strip('\n')
                line = line.replace('q_j: ', '')
                q_j.append(line)

    history_loss = list(map(float, history_loss))
    select_times = list(map(float, select_times))
    q_j = list(map(float, q_j))
    print(history_loss)
    print(select_times)
    print(q_j)
    OPTIONERS = user_index  # 备选
    WINNERS = []  # 最终
    # 冒泡排序；按照单位精度成本给OPTIONERS排序 贝塔值
    n = len(OPTIONERS)
    if n > 0:  # OPTIONERS 不为空#real cost-》c  用户上报 # real acc-》q_j
        for i in range(n):
            for j in range(0, n - i - 1):
                if Cost[j] / q_j[j] > Cost[j + 1] / q_j[j + 1]:
                    OPTIONERS[j], OPTIONERS[j + 1] = OPTIONERS[j + 1], OPTIONERS[j]
    print(OPTIONERS)

    total = 0
    k = 0
    Budget = 20
    for i in range(0, len(OPTIONERS)):
        total = total + q_j[j]
        if (Budget / total >= Cost[i] / q_j[i]):
            k = i + 1
    print('k=', k)

    PAYMENTS = []
    SELLERS = user_index
    for i in range(0,k):
        WINNERS.append(OPTIONERS[i])

    # 若k = |S|
    if len(WINNERS) == len(SELLERS):
        Sigma = 0  # 贝塔
        for i in SELLERS:
            Sigma = Sigma + q_j[i]
        for i in WINNERS:
            Payment = Budget * q_j[i] / Sigma
            PAYMENTS.append(Payment)  # 每一个winner 都有payment
    # 若k < |S|
    if len(WINNERS) < len(SELLERS):
        n = len(WINNERS)
        Sigma = 0
        for i in SELLERS:
            Sigma = Sigma + q_j[i - 1]
        if Budget / Sigma < Cost[n] / q_j[n]:
            for i in range(0, len(WINNERS)):
                Payment = Budget * q_j[i] / Sigma
                PAYMENTS.append(Payment)
        else:
            for i in range(0, len(WINNERS)):
                Payment = Cost[n] / q_j[n]
                PAYMENTS.append(Payment)
    print(PAYMENTS)
    idxs_users = np.array(WINNERS)
    # Training
    current_loss = []
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0




    ### number of users participating in the training round
    # m = max(int(args.frac * args.num_users), 1) 
    m = args.num_users #default is 5

    ### randomly choose that many users to participate
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
    #idxs_users = np.array(WINNERS)

    print('user indexes are:', idxs_users,'type', type(idxs_users)) # print check
    '''
    acc = []
    Rn= [0]
    u_30=[]
    Budget = 20
    for epoch in tqdm(range(args.epochs)):
        user_number = 10
        user_index = []
        Cost = []
        current_loss = []
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0
        history_q = []
        history_loss = []
        select_times = []
        q_j = []

        for index in range(len(user_groups)):
            user_index.append(index + 1)
            Cost.append(len(user_groups[index + 1]) / 1000)

        with open('../save/MRfed_{}_{}_{}_{}_update.txt'.format(args.dataset, args.model, epoch, 2), 'r') as f:
            for line in f.readlines():
                if re.match('Loss *', line) != None:
                    line = line.strip('\n')
                    line = line.replace('Loss value: ', '')
                    history_loss.append(line)
                elif re.match('Select *', line) != None:
                    line = line.strip('\n')
                    line = line.replace('Select times: ', '')
                    select_times.append(line)
                elif re.match('q_j *', line) != None:
                    line = line.strip('\n')
                    line = line.replace('q_j: ', '')
                    q_j.append(line)
                elif re.match('history_q *', line) != None:
                    line = line.strip('\n')
                    line = line.replace('history_q: ', '')
                    history_q.append(line)

        history_loss = list(map(float, history_loss))
        select_times = list(map(float, select_times))
        q_j = list(map(float, q_j))
        history_q = list(map(float, history_q))

        print(history_loss)
        print(select_times)
        print(q_j)

        #OPTIONERS = []  # 备选
        WINNERS = []  # 最终

        #OPTIONERS=user_index
        # 冒泡排序；按照单位精度成本给OPTIONERS排序 贝塔值
        OPTIONERS = user_index
        n = user_number
        if n > 0:  # OPTIONERS 不为空#real cost-》c  用户上报 # real acc-》q_j
            for i in range(n):
                for j in range(0, n - i - 1):
                    if Cost[j] / q_j[j] > Cost[j + 1] / q_j[j + 1]:
                        OPTIONERS[j], OPTIONERS[j + 1] = OPTIONERS[j + 1], OPTIONERS[j]
        print(OPTIONERS)
        '''
        # 冒泡排序；按照单位精度成本给OPTIONERS排序 贝塔值
        OPTIONERS = user_index
        n = user_number
        if n > 0:  # OPTIONERS 不为空#real cost-》c  用户上报 # real acc-》q_j
            for i in range(n):
                for j in range(0, n - i - 1):
                    if Cost[j] / q_j[j] > Cost[j + 1] / q_j[j + 1]:
                        OPTIONERS[j], OPTIONERS[j + 1] = OPTIONERS[j + 1], OPTIONERS[j]
        print(OPTIONERS)
        '''
        total = 0
        k = 0
        Budget += 0.5
        for i in range(0, len(OPTIONERS)):
            total = total + q_j[j]
            if (Budget / total >= Cost[i] / q_j[i]):
                k = i + 1
        print('k=', k)
        #k=5
        PAYMENTS = []
        SELLERS = user_index

        for i in range(0, k):
            WINNERS.append(OPTIONERS[i])

        # 若k = |S|
        if len(WINNERS) == len(SELLERS):
            Sigma = 0  # 贝塔
            for i in SELLERS:
                Sigma = Sigma + q_j[i-1]
            for i in WINNERS:
                Payment = Budget * q_j[i-1] / Sigma
                PAYMENTS.append(Payment)  # 每一个winner 都有payment
        # 若k < |S|
        if len(WINNERS) < len(SELLERS):
            n = len(WINNERS)
            Sigma = 0
            for i in SELLERS:
                Sigma = Sigma + q_j[i - 1]
            if Budget / Sigma < Cost[n] / q_j[n]:
                for i in range(0, len(WINNERS)):
                    Payment = Budget * q_j[i] / Sigma
                    PAYMENTS.append(Payment)
            else:
                for i in range(0, len(WINNERS)):
                    Payment = Cost[n] / q_j[n]
                    PAYMENTS.append(Payment)
        print(PAYMENTS)

        ### number of users participating in the training round
        # m = max(int(args.frac * args.num_users), 1)
        m = args.num_users  # default is 5
        idxs_users = np.array(WINNERS)
        #idxs_users = random.sample(range(1, 11), 5)
        ### randomly choose that many users to participate

        if epoch==49:
            u_30=np.array(WINNERS)
        if epoch>=50:
            idxs_users = np.array(u_30)\


        print('user indexes are:', idxs_users, 'type', type(idxs_users))  # print check
        # List of fraction of data from each user
        total_data = sum(len(user_groups[i]) for i in range(1, m + 1))
        fraction = [len(user_groups[i]) / total_data for i in range(1, m + 1)]
        #     # print("data fraction is:", fraction) # print check

        # Training
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        current_loss=[]

        # note that the keys for train_dataset are [1,2,3,4,5]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx],
                                      idxs=user_groups[idx], logger=logger)
            w, loss ,c= local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch,local_round=1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            current_loss.append(c)

        # update global weights
        global_weights = average_weights(local_weights, fraction)
        # print("global_weights is", global_weights) # print check




        # # test if the functions are behaving correctly
        # average_gradients = average_weights(gradients, fraction) # average_weights can be used to average gradients
        # global_weights2 = update_weights_from_gradients(average_gradients, global_model.state_dict()) 
        # print("global_weights2 is", global_weights) # print check

        # update global weights
        global_model.load_state_dict(global_weights) ### update the 2^n submodels as well 

        loss_avg = sum(local_losses) / len(local_losses)



        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch. 
        # For this case, since all users are participating in training, we need to adjust the code
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users): (this doesn't apply in our case)
        # for idx in idxs_users:
        #     local_model = LocalUpdate(args=args, dataset=train_dataset[idx],
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {train_loss}') # print check 
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        Z=dict(zip(WINNERS,current_loss))
        '''
        for i, item in enumerate(WINNERS):
            k = history_loss[item - 1] - z[item - 1]
            if k < 0.000000:
                history_loss[item - 1] = 0.0
            else:
                history_loss[item - 1] = k
        '''
        for item in Z:
            k = history_loss[item - 1] - Z[item]
            if k < 0.0:
                history_loss[item - 1] = 0.0
            else:
                history_loss[item - 1] = k

        print('history_loss', history_loss)

        # 计算q_b：1/0
        t = 0
        for users in history_q:
            history_q[t] = (history_q[t] * select_times[t] + history_loss[t]) / (select_times[t] + 1)
            t += 1
        print('history_q', history_q)

        # 计算q_j
        #i = 1  # 轮次
        u = []
        q_j_new = []
        k = 0
        if epoch + 1 == 1:
            for users in select_times:
                u.append(0)
        elif epoch + 1 > 1:
            for users in select_times:
                u.append(((3 * np.log(epoch + 1)) / (2 * select_times[k])) ** 0.5)
                k += 1
        print(u)
        q = 0
        for users in history_q:
            q_j_new.append(min(history_q[q] + u[q], 1))
            q += 1
        print('q_j', q_j_new)

        # 计算Snm_t
        select_user = WINNERS
        for t in select_user:
            select_times[t - 1] = select_times[t - 1] + 1
        print('select_times', select_times)

        select_user = WINNERS
        Q = 0
        for t in select_user:
            Q = Q + history_q[t - 1]

        Rn.append(Rn[epoch]+Q)
        print('select_times', select_times)

        update_loss = history_loss
        t = 0
        for i in current_loss:
            if current_loss[t] > 1:
                current_loss[t] = 1
                t += 1

        for i, item in enumerate(current_loss):
            update_loss[i] = current_loss[i]

        # write information into a file
        accuracy_file = open('../save/MRfed_{}_{}_{}_{}_update.txt'.format(args.dataset, args.model,
                                                                              epoch+1, 2), 'a')
        # for subset in powerset:
        #    accuracy_lines = ['Trainset: '+args.traindivision+'_'+''.join([str(i) for i in subset]), '\n',
        #            'Accuracy: ' +str(accuracy_dict[subset]), '\n',
        #            '\n']
        #    accuracy_file.writelines(accuracy_lines)
        print(current_loss)
        for key, item in enumerate(history_loss):
            shapley_lines = ['Data contributor: ' + str(key + 1), '\n',
                             'Loss value: ' + str(update_loss[key]), '\n',
                             'Select times: ' + str(select_times[key]), '\n',
                             'q_j: ' + str(q_j_new[key]), '\n',
                             'history_q: ' + str(history_q[key]), '\n',
                             '\n']
            accuracy_file.writelines(shapley_lines)
        accuracy_file.close()
    # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        acc.append(test_acc)
        print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    plt.show()
    print(acc)
    print(Rn)
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy')
    plt.plot(range(len(acc)), acc, color='k')
    plt.ylabel('Accuracy')
    plt.xlabel('Global Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    plt.show()
