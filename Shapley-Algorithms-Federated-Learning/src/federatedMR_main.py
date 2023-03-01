#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import math
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import re
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP
from utils import get_dataset, average_weights, exp_details, powersettool, shapley

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time = time.time() # start the timer

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
        #print(img_size) # print check

        len_in = 1
        for x in img_size:
            len_in *= x
        # print('number of dimension_in is :', len_in) # print check
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)    
    # else:
    #     exit('Error: unrecognized model')

    ### number of users participating in the training round
    #m = max(int(args.frac * args.num_users), 1)
    m = args.num_users #default is 5
    # Powerset list (except nullset and the set itself)
    powerset = list(powersettool(range(1,m+1))) #generate a powerset list of tuples eg [(),(1,),(2),(1,2)]

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()    ### create 2^5 submodels based on this 

    # Initialize all the sub-models
    submodel_dict = {}
    for subset in powerset[1:-1]: #exclude the null and global set 
        submodel_dict[subset] = copy.deepcopy(global_model)
        submodel_dict[subset].to(device)
        submodel_dict[subset].train()

    # dictionary containing all the accuracies for the subsets    
    accuracy_dict = {}
    accuracy_dict[()] = 0



    # get users' history shapley value
    history_shapley = []
    with open('../save/MRfed_OurMNIST_mlp_5_4_old.txt', 'r') as f:
        for line in f.readlines():
            if re.match('Shapley *', line) != None:
                line = line.strip('\n')
                line = line.replace('Shapley value: ', '')
                history_shapley.append(line)
    history_shapley=list(map(float,history_shapley))
    print(history_shapley)


    class KandStma:
        def __init__(self, k, sita):
            self.k = k
            self.sita = sita
            self.ratio = k / sita
            # print(self.k)
            # print(self.sita)
            # print(self.ratio)


    # 获得S集合中所有元素的k/sita之和
    def rt_sum_list_s():
        answer = 0.
        for s in list_s:
            answer += s.ratio
        return answer


    # 判断是否满足加入集合S的条件
    def if_add_to_s(ii):
        ratio = list_kAndSita[ii].ratio
        sum_s = rt_sum_list_s()
        return ratio < (ratio + sum_s) / len(list_s)


    #####################################################################
    # 计算聚合器的效用函数
    # 计算Xi
    def compute_xi(si):
        n = len(list_s) - 1
        sum_s = rt_sum_list_s()  # 所有k/sita之和
        xi = n / (si.sita * sum_s) * (1 - n * si.ratio / sum_s)
        return xi


    # 计算Y
    def compute_y(r):
        count = 1.0
        for si in list_s:  ## list_s是ratio列表
            xi = compute_xi(si)
            count += math.log(1 + si.sita * xi * r)
        return count


    def fun_r(r):  # 聚合器效用ψ
        y = compute_y(r)
        return lamda * math.log(y) - r


    def dfun_r(r):  ##聚合器效用一阶导数
        sum1 = 0.0
        for si in list_s:
            xi = compute_xi(si)
            sum1 = sum1 + si.sita * xi / (1 + si.sita * xi * r)
        return 1 - lamda * sum1 / compute_y(r)


    # 计算ti并打印 number of participation rounds of client i
    def fun_ti(r):
        for ii in list_s:
            ti.append(round(compute_xi(ii) * r * unit))
        return ti


    # 总成本
    def fun_tci():
        t = 0
        for i in list_s_index:
            tci.append(ti[t] * ki[t])
            t+=1
        return tci


    # 计算Ui 客户i的效用函数
    def fun_ui(r):
        sumtt = 0
        i = 0

        for s in list_s:
            sumtt += s.sita * ti[i]
            i += 1
        i = 0
        for s in list_s:
            ui.append(s.sita * ti[i] / sumtt * r - ti[i] / 10 * s.k)
            i += 1
        return ui
    # Computation of the NE
    list_kAndSita = []
    ki = []  # 训练成本
    ti = []  # 轮次
    tci = []
    ui = []
    unit = 10
    for i in range(100):
        ki.append(random.uniform(1, 2))  # 随机1-2的值进行模拟
        list_kAndSita.append(KandStma(ki[i], history_shapley[i]))

    # 新增_start
    list_s_index = []
    list_s = []
    theta_th = 0.0  # theta的阈值
    i = 0

    while len(list_s) < 2 and i < len(list_kAndSita):
        if list_kAndSita[i].sita >= theta_th:
            list_s.append(list_kAndSita[i])
            list_s_index.append(i)
        i = i + 1
    while i < len(list_kAndSita)  and if_add_to_s(i) and len(list_s_index) < m:#5为用户数量
        if list_kAndSita[i].sita >= theta_th:
            list_s.append(list_kAndSita[i])
            list_s_index.append(i)
        i = i + 1
    print('s 集合的大小为：%d' % len(list_s))
    list_s_index=[i+1 for i in list_s_index]
    a=train_dataset.values()
    b=user_groups.values()
    train_dataset=dict(zip(list_s_index,a))
    user_groups=dict(zip(list_s_index,b))


    # 求R（Reward）
    lamda = 20  # 系统参数原文中置为10
    r0 = 0  # ψ
    eta = 0.01  # 步长
    epsilon = 0.00000001  # 阈值，当梯度小于阈值时，结束
    history_r0 = [r0]
    while True:
        gradient = dfun_r(r0)  # 通过梯度下降法逼近最优值
        last_r0 = r0
        r0 = r0 - eta * gradient
        history_r0.append(r0)
        if abs(fun_r(last_r0) - fun_r(r0)) < epsilon:
            break
    user_round = []
    user_round = fun_ti(r0)

    print("训练成本：", fun_tci())
    i=0
    for i in range(m):
        if user_round[i]<=0:
            user_round[i]=1
    print(user_round)
    # Training

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    ### choose that many users to participate
    idxs_users = np.array(list_s_index)
    #idxs_users = np.arange(1, m+1)
    print('user indexes are:', idxs_users,'type', type(idxs_users)) # print check


    # List of fraction of data from each user
    total_data = sum(len(train_dataset[i]) for i in train_dataset.keys())
    fraction = [len(train_dataset[i])/total_data for i in train_dataset.keys()]
    # print("data fraction is:", fraction) # print check 

    # dictionary of shapley for all rounds
    shapley_dict = {}
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')
        global_model.train()
        '''
        choose_round(history_shapley)
        '''
        # The round that the user chooses to participate in
        i=0
        # note that the keys for train_dataset are [1,2,3,4,5]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx],idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,local_round=user_round[i])
            i+=1
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights, fraction)
        # update global weights
        global_model.load_state_dict(global_weights) ### update the 2^n submodels as well 

        loss_avg = sum(local_losses) / len(local_losses)

        train_loss.append(loss_avg)

        # update sub-model weights
        for subset in powerset[1:-1]:
            # for MR algorithm, we only need to average the subset of weights
            subset_weights = average_weights([local_weights[i-1] for i in subset], [fraction[i-1] for i in subset])
            submodel_dict[subset].load_state_dict(subset_weights)

        # Calculate avg training accuracy over all users at every epoch. 
        # For this case, since all users are participating in training, we need to adjust the code
        list_acc, list_loss = [], []
        global_model.eval()
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
            print(f' \nAvg Training Stats after {epoch} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # Test inference after every round
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {epoch} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        accuracy_dict[powerset[-1]] = test_acc
        train_accuracy.append(test_acc)

        # Test inference for the sub-models in submodel_dict
        for subset in powerset[1:-1]: 
            test_acc, test_loss = test_inference(args, submodel_dict[subset], test_dataset)
            # print(f' \n Results after {args.epochs} global rounds of training:')
            # print("|---- Test Accuracy for {}: {:.2f}%".format(subset, 100*test_acc))
                
            accuracy_dict[subset] = test_acc

        shapley_dict_add = shapley(accuracy_dict, args.num_users)


        i=1
        for idx in idxs_users :
            if shapley_dict.get(idx):
                shapley_dict[idx].append(shapley_dict_add[i])
                i+=1
            else:
                shapley_dict[idx] = [shapley_dict_add[i]]
                i+=1


    # Saving the objects train_loss and train_accuracy:
    #file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #           args.local_ep, args.local_bs)

    #with open(file_name, 'wb') as f:
    #    pickle.dump([train_loss, train_accuracy], f)

    totalRunTime = time.time()-start_time
    print('\n Total Run Time: {0:0.4f}'.format(totalRunTime))  # print total time

    #caculate the final shapley
    aggregate_list=[]
    keys=shapley_dict.keys()
    values = shapley_dict.values()
    for value in values:
        shapley_sum1=0
        shapley_sum2=0
        decay=0.5
        for i in range(0,args.epochs):
             shapley_sum1+=(value[i])
            # shapley_sum2+=(value[i])*math.pow(decay,i+1)
        for i in range(0, args.epochs):
             aggregate_list.append(value[i] / shapley_sum1)

             #print(shapley_sum1)
             #print(shapley_sum2)
        #aggregate_list.append(shapley_sum2/shapley_sum1)
    i=0
    for key in keys:
        shapley_dict[key].append(aggregate_list[i])
        i+=1
    print(shapley_dict)
    t=0
    for i in idxs_users:
        history_shapley[i-1]=aggregate_list[t]
        t+=1
    user = []
    for i in range(1, 101):
        user.append(i)
    update_shapley = dict(zip(user, history_shapley))
    #write information into a file 
    accuracy_file = open('../save/MRfed_{}_{}_{}_{}_update.txt'.format(args.dataset, args.model,
                            args.epochs, args.traindivision), 'a')
    #for subset in powerset:
    #    accuracy_lines = ['Trainset: '+args.traindivision+'_'+''.join([str(i) for i in subset]), '\n',
    #            'Accuracy: ' +str(accuracy_dict[subset]), '\n',
    #            '\n']
    #    accuracy_file.writelines(accuracy_lines)

    for key in update_shapley:
        shapley_lines = ['Data contributor: '+str(key),'\n',
                'Shapley value: '+ str(update_shapley[key]), '\n',
                '\n']
        accuracy_file.writelines(shapley_lines)
    #lines = ['Total Run Time: {0:0.4f}'.format(totalRunTime),'\n']
    #accuracy_file.writelines(lines)
    #accuracy_file.close()

    #with open('../save/MRfed_{}_{}_{}_{}.pkl'.format(args.dataset, args.model,
    #                    args.epochs, args.traindivision), 'wb') as f:
    #    pickle.dump(accuracy_dict, f)
    print(train_accuracy)

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    print(train_loss)
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                 format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs))
    plt.show()

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    plt.show()

'''
    # write information into a file
accuracy_file = open('../save/test.txt', 'a')
for key in range (1,101):
    shapley_lines = ['Data contributor: ' + str(key), '\n',
                         'Shapley value: ' + '['+str(0.5)+']', '\n',
                         '\n']
    accuracy_file.writelines(shapley_lines)
accuracy_file.close()


'''