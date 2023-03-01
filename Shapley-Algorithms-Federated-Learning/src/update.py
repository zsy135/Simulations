#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, and test (function as validation) (90, 10)
        # idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_train = idxs[:]
    
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))] ## don't need this line any more
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # print('length of trainloader is', len(trainloader.dataset)) # print check 

        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)

        # print('length of validloader is', len(validloader.dataset)) # print check
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=Fal'se)
        # print('length of testloader is', len(testloader.dataset)) # print check
        return trainloader


    def update_weights(self, model, global_round, local_round, main=1):
        '''
        Note that main==1 is when the main model (trained on all the data) is used
        when main==0, this function is much less verbose to reduce time taken
        '''

        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(local_round):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()   
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                if self.args.verbose and (batch_idx % 170 == 0) and (main==1):
                    print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                        global_round, iter,  loss.item()))
                    if iter==local_round-1:
                        user_loss = loss.item()
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), user_loss

    # def inference(self, model):
    #     """ Returns the inference accuracy and loss.
    #     """
        
    #     model.eval()
    #     loss, total, correct = 0.0, 0.0, 0.0

    #     for batch_idx, (images, labels) in enumerate(self.testloader):
    #         images, labels = images.to(self.device), labels.to(self.device)
            
    #         # print('length of images is', len(images), 'type:', type(images), 'shape:', images.shape) # print check 
    #         # Inference
    #         outputs = model(images)
    #         batch_loss = self.criterion(outputs, labels)
    #         loss += batch_loss.item()

    #         # Prediction
    #         _, pred_labels = torch.max(outputs, 1)
    #         pred_labels = pred_labels.view(-1)
    #         # print(pred_labels) # print check 
    #         # print('correct is', labels) # print check 
    #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #         total += len(labels)
    #         # print('total is', total) # print check 

    #     accuracy = correct/total
    #     # print('accuracy is', accuracy) # print check 
    #     return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss