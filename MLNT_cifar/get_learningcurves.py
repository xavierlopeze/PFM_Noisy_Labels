import wandb
import sys

# Pytorch libraries
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Internal files
import config
import dataloader
import models
# from baseline import get_model, save_checkpoint

import math
import os
import sys
import time
from collections import OrderedDict
import random

import pandas as pd

def get_model():

    # Get model from config
    if config.model == "resnet18":
        model = models.resnet18(pretrained=config.pretrained)
    elif config.model == "resnet34":
        model = models.resnet34(pretrained=config.pretrained)
    elif config.model == 'resnet50':
        model = models.resnet50(pretrained=config.pretrained)
    elif config.model == "resnet101":
        model = models.resnet101(pretrained=config.pretrained)
    elif config.model == "resnet152":
        model = models.resnet152(pretrained=config.pretrained)
    elif config.model == "resnext50_32x4d":
        model = models.resnet34(pretrained=config.pretrained)
    elif config.model == 'resnext101_32x8d':
        model = models.resnet50(pretrained=config.pretrained)
    elif config.model == "wide_resnet50_2":
        model = models.resnet101(pretrained=config.pretrained)
    elif config.model == "wide_resnet101_2":
        model = models.resnet152(pretrained=config.pretrained)
    else:
        raise ValueError('%s not supported'.format(config.model))

    # Initialize fc layer
    (in_features, out_features) = model.fc.in_features, model.fc.out_features
    model.fc = torch.nn.Linear(in_features, out_features)
    return model

def load_checkpoint(epoch, netwk):
  if netwk == "baseline":
    checkpoint = torch.load(config.drive_dir + '/checkpoint/' + config.checkpoint + "_" + str(epoch) + '.pth.tar')
  elif ntwk == "student":
    checkpoint = torch.load(config.drive_dir + '/checkpoint/' + config.id + "_student_" + str(epoch) + '.pth.tar')
  elif ntwk == "teacher":
    checkpoint = torch.load(config.drive_dir + '/checkpoint/' + config.id + "_teacher_" + str(epoch) + '.pth.tar')

  net.load_state_dict(checkpoint['state_dict'])

def test(loader):
    net.eval()
    # test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = net(inputs).cuda()

        # test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Grab validation results
    test_acc = 100. * correct/total

    return(round(float(test_acc),2))


def get_accy_list( max_epoch, loader, netwk):
  max_epoch = 120

  accy_list = []
  for epoch in range(1,max_epoch):
    load_checkpoint(epoch, netwk)
    accy_list.append(test(loader))
    print('status: ' + str(len(accy_list)*100/max_epoch) + '%')
  return(accy_list)




#Get baseline
for ntwk in ["baseline", "teacher", "student"]:

    # Get the original_dataset
    loader = dataloader.KeyDataLoader()
    train_loader, valid_loader, test_loader = loader.run()

    # Hyper Parameter settings
    random.seed(config.seed)
    # torch.cuda.set_device(config.gpuid)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    use_cuda = torch.cuda.is_available()

    # Networks setup
    print('\nModel setup')
    print('| Building network: {}'.format(config.model))
    net = get_model()
    if use_cuda:
      net.cuda()

    #Computing accys in lists
    print("computing test list")
    test_list  = get_accy_list(config.num_epochs, test_loader, ntwk)
    print(test_list)
    print("\ncomputing val list")
    valid_list = get_accy_list(config.num_epochs, valid_loader,ntwk)
    print(valid_list)
    print("\ncomputing train list\n")
    train_list = get_accy_list(config.num_epochs, train_loader,ntwk)

    #Append the accys in a dataframe
    df = pd.DataFrame()
    df["epoch"] = range(1,config.num_epochs)
    df["test"] = test_list
    df["valid"] = valid_list
    df["train"] = train_list

    #Export the dataframe
    if ntwk == "baseline":
      clean_kv = config.drive_dir +"/learning_curves/"+ config.checkpoint+".csv"
    elif ntwk == "student":
      clean_kv = config.drive_dir +"/learning_curves/"+ config.id+"_student.csv"
    elif ntwk == "teacher":
      clean_kv = config.drive_dir +"/learning_curves/"+ config.id+"_teacher.csv"

    df.to_csv(clean_kv, sep=';', index=False,header = True)
