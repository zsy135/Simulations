#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import OurMNIST
# from sampling import cifar_iid, cifar_noniid
import pickle
import numpy as np
import \
    random  # used to shuffle lists of data from dictionary, since our dictionary is too organised (1 then 2 then 3...)
from itertools import chain, combinations
from scipy.special import comb
from update import test_inference  # used in TMC to get the score of the model

random.seed(0)  # used a random seed for reproducibility


class OurMNIST(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        data: list of (image_tensor, label) tuples.
        transform (callable, optional): optional tranform to be applied on a sample.
        """
        self.data = data

    def __getitem__(self, index):
        img, target = self.data[index][0], self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    # if args.dataset == 'cifar':
    #     print('cifar')
    #     data_dir = '../data/cifar/'
    #     apply_transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #     train_dataset = datasets.MNIST(data_dir, train=True, download=True,
    #                                    transform=apply_transform)

    #     test_dataset = datasets.MNIST(data_dir, train=False, download=True,
    #                                   transform=apply_transform)

    #     # sample training data amongst users
    #     if args.iid:
    #         # Sample IID user data from Mnist
    #         user_groups = cifar_iid(train_dataset, args.num_users)
    #     else:
    #         # Sample Non-IID user data from Mnist
    #         if args.unequal:
    #             # Chose uneuqal splits for every user
    #             raise NotImplementedError()
    #         else:
    #             # Chose euqal splits for every user
    #             user_groups = cifar_noniid(train_dataset, args.num_users)

    if args.dataset == 'OurMNIST' and args.trueshapley == '' and args.num_users == 5:
        data_dir = '../data/OurMNIST/'

        # The individual trainset according to the argument passed
        # traindivision_index will be 'a' where a is the dictionary
        traindivision_index = args.traindivision

        with open(data_dir + 'dictionary' + traindivision_index[0] + '.out', 'rb') as pickle_in:
            dictionary = pickle.load(pickle_in)  # change name
            # print("type of dictionary: ", type(dictionary)) # print check

        # print(dictionary['trainset_1_1'][0][0].shape) # print check

        # Form the train_dataset, which is a dictionary with (user indexes: data)
        # user_groups is a dictionary with (user indexes: data indexes)
        train_dataset = {}
        user_groups = {}
        num_users = args.num_users
        for i in range(1, num_users + 1):
            # Form train_dataset[1] to train_dataset[5]
            train_dataset[i] = OurMNIST(random.sample(dictionary['trainset_' + traindivision_index[0] + '_' + str(i)],
                                                      len(dictionary[
                                                              'trainset_' + traindivision_index[0] + '_' + str(i)])))
            user_groups[i] = [j for j in range(len(train_dataset[i]))]

            # print(len(train_dataset[i])) # print check

        # print('dictionary '+traindivision_index+' is used') # print check

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=apply_transform)

        return train_dataset, test_dataset, user_groups  # note that train_dataset for this case is a dictionary

    elif args.dataset == 'OurMNIST' and args.trueshapley == '':
        data_dir = '../data/OurMNIST/'

        # The individual trainset according to the argument passed
        # traindivision_index will be 'a' where a is the dictionary
        traindivision_index = args.traindivision

        with open(data_dir + 'dictionary' + traindivision_index[0] + '.out', 'rb') as pickle_in:
            dictionary = pickle.load(pickle_in)  # change name
            # print("type of dictionary: ", type(dictionary)) # print check

        # print(dictionary['trainset_1_1'][0][0].shape) # print check

        # Form the train_dataset, which is a dictionary with (user indexes: data)
        # user_groups is a dictionary with (user indexes: data indexes)
        train_dataset = {}
        user_groups = {}
        num_users = args.num_users

        for i in range(1, int(num_users / 2) + 1):
            # Form train_dataset[1] to train_dataset[10]
            length = len(dictionary['trainset_' + traindivision_index[0] + '_' + str(i)])
            train_dataset[2 * i - 1] = OurMNIST(
                random.sample(dictionary['trainset_' + traindivision_index[0] + '_' + str(i)],
                              length))
            user_groups[2 * i - 1] = [j for j in range(length) if j % 2 == 0]

            train_dataset[2 * i] = copy.deepcopy(train_dataset[2 * i - 1])
            user_groups[2 * i] = [j for j in range(length) if j % 2 == 1]

        '''t=1
        for i in range(1, int(num_users/20 +1)):
            # Form train_dataset[1] to train_dataset[10]
            length=len(dictionary['trainset_' + traindivision_index[0] + '_' + str(i)])
            for j in range(1,11):
                train_dataset[2 * t - 1] = OurMNIST(random.sample(dictionary['trainset_' + traindivision_index[0] + '_' + str(i)],length))

                user_groups[2 * t - 1] = [int(j/10) for j in range(length) if j % 20 == 0]

                train_dataset[2 * t] = copy.deepcopy(train_dataset[2 * t - 1])
                user_groups[2 * t] = [int(j/10+1)  for j in range(length) if j % 20 == 1]
                t+=1

            # print(len(user_groups[2*i-1])) # print check
            # print(len(user_groups[2*i])) # print check
        '''
        # print('dictionary '+traindivision_index+' is used') # print check

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=apply_transform)

        return train_dataset, test_dataset, user_groups  # note that train_dataset for this case is a dictionary

    elif args.dataset == 'OurMNIST' and args.trueshapley != '' and args.num_users == 5:
        data_dir = '../data/OurMNIST/'

        # The individual trainset according to the argument passed
        # traindivision_index will be 'a' where a is the dictionary
        traindivision_index = args.traindivision
        combination_index = args.trueshapley
        if len(combination_index) == 1:
            with open(data_dir + 'dictionary' + traindivision_index + '.out', 'rb') as pickle_in:
                dictionary = pickle.load(pickle_in)  # change name
                name_trainset = 'trainset_' + traindivision_index + '_' + combination_index
                # print(name_trainset) # print check
                # print('dictionary is', dictionary['trainset_'+traindivision_index+'_1'][0][0].shape) # print check
                # print("type of dictionary: ", type(dictionary)) # print check

        else:
            with open(data_dir + 'cdictionary' + traindivision_index + '.out', 'rb') as pickle_in:
                dictionary = pickle.load(pickle_in)  # change name
                name_trainset = 'ctrainset_' + traindivision_index + '_' + combination_index
                # print(name_trainset) # print check
                # print('cdictionary is', dictionary['ctrainset_'+traindivision_index'_12'][0][0].shape) # print check

        user_groups = {}

        train_dataset = OurMNIST(random.sample(dictionary[name_trainset],
                                               len(dictionary[name_trainset])))

        # print(len(train_dataset)) # print check

        # print('dictionary'+traindivision_index+'_'+combination_index +' is used') # print check

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=apply_transform)

        return train_dataset, test_dataset, user_groups  # note that train_dataset for this case is just datasets objects  


    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups



def average_weights(w, fraction):  # this can also be used to average gradients
    """
    :param w: list of weights generated from the users
    :param fraction: list of fraction of data from the users
    :Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0]) #copy the weights from the first user in the list 
    for key in w_avg.keys():
        w_avg[key] *= (fraction[0]/sum(fraction))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (fraction[i]/sum(fraction))
        # w_avg[key] = torch.div(w_avg[key], len(w)) # this is wrong implementation since datasets can be unbalanced
    return w_avg 

def calculate_gradients(new_weights, old_weights):
    """
    :param new_weights: list of weights generated from the users
    :param old_weights: old weights of a model, probably before training
    :Returns the list of gradients.
    """

    gradients = []
    for i in range(len(new_weights)):
        gradients.append(copy.deepcopy(new_weights[i]))
        for key in gradients[i].keys():
            gradients[i][key] -= old_weights[key]
    return gradients

def update_weights_from_gradients(gradients, old_weights):
    """
    :param gradients: gradients
    :param old_weights: old weights of a model, probably before training
    :Returns the updated weights calculated by: old_weights+gradients.
    """
    updated_weights = copy.deepcopy(old_weights)
    for key in updated_weights.keys():
        updated_weights[key] = old_weights[key] + gradients[key]
    return updated_weights
    


def powersettool(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(1,N+1):
        shapley_dict[i] = 0
    for key in utility:
        if key != ():
            for contributor in key:
                # print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i!=contributor)]
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution /((comb(N-1,len(key)-1))*N)
    return shapley_dict


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Dictionary Number   : {args.traindivision}\n')
    print(f'    Global Rounds   : {args.epochs}\n')
    
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    # print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
