# Pytorch libraries
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Internal files
import config
import dataloader
import models
from baseline import get_model, save_checkpoint

import math
import os
import sys
import time
from collections import OrderedDict
import random


def scheduler(epoch: int):
    global lr
    lr = config.lr
    if epoch > config.start_epoch:
        lr = lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0
    scheduler(epoch)
    for step, (inputs, targets) in enumerate(train_loader):
        init_time = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)

        class_loss = criterion(outputs, targets)
        class_loss.backward(retain_graph=True)

         if step > config.start_iter or epoch > 1:
        #if step > 0 or epoch > 0:

            if step > config.mid_iter or epoch > 1:
                config.eps = 0.999
                alpha = config.alpha
            else:
                u = (step - config.start_iter)/(config.mid_iter - config.start_iter)
                alpha = config.alpha*math.exp(-5*(1-u)**2)

            if init:
                init = False
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.mul_(config.eps).add_((1-config.eps), param.data)

            _, feats = pretrain_net(inputs, get_feat=True)
            tch_outputs = tch_net(inputs, get_feat=False)
            p_tch = F.softmax(tch_outputs, dim=1)
            p_tch.detach_()

            for i in range(config.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                for n in range(int(targets.size(0)*config.perturb_ratio)):
                    num_neighbor = 10
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1, feat.size(0))
                    feat.data = feat.data.expand(targets.size(0), feat.size(0))
                    dist = torch.sum((feat-feats)**2, dim=1)
                    _, neighbor = torch.topk(dist.data, num_neighbor+1, largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1, num_neighbor)]]

                fast_loss = criterion(outputs, targets_fast)

                grads = torch.autograd.grad(fast_loss, net.parameters(),
                                            create_graph=False,
                                            retain_graph=True,
                                            only_inputs=True)

                fast_weights = OrderedDict(
                    (name, param - config.meta_lr*grad)
                    for ((name, param), grad) in zip(net.named_parameters(), grads))

                fast_out = net.forward(inputs,fast_weights)

                logp_fast = F.log_softmax(fast_out,dim=1)
                consistent_loss = consistent_criterion(logp_fast, p_tch)
                consistent_loss = consistent_loss*alpha/config.num_fast
                consistent_loss.backward()

        optimizer.step()

        # train_loss += class_loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # Grab training results
        print("| Epoch: {}/{}, step: {}/{}, loss: {:.3f}, acc: {:.3f},"
              " time: {:.3f}".format(epoch,
                                     config.num_epochs,
                                     step + 1,
                                     len(train_loader.dataset) // config.batch_size + 1,
                                     class_loss.data.item(),
                                     100. * correct / total,
                                     time.time() - init_time),
              end="\r")


def valid(epoch, network):
    global best_acc
    network.eval()
    # val_loss = 0
    correct = 0
    total = 0
    for step, (inputs, targets) in enumerate(valid_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = network(inputs)
            loss = criterion(outputs, targets)

        # valid_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # Grab validation results
    valid_acc = 100. * correct / total
    valid_results = ("| Epoch: {}/{}, val_loss: {:.3f}, val_acc: {:.3f}, "
                     "lr: {:.6f}".format(epoch,
                                         config.num_epochs,
                                         loss.data.item(),
                                         valid_acc,
                                         lr))

    # Save checkpoint when best model
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('| Saving Best Model ...', end="\r")
        save_point = './checkpoint/%s.pth.tar' % config.id
        save_checkpoint({
            'state_dict': network.state_dict(),
            'best_acc': best_acc,
        }, save_point)

    return valid_results


def test():
    test_net.eval()
    # test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valid_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = test_net(inputs)
            loss = criterion(outputs, targets)

        # test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Grab validation results
    test_acc = 100. * correct/total
    test_results = "| test_loss: {:.3f}, test_acc: {:.3f}".format(
        loss.data.item(), test_acc)
    record.write(test_results)
    record.flush()

    print(test_results)


if __name__ == '__main__':

    # Checkpoint dir.
    record = open('./checkpoint/' + config.checkpoint + '_test.txt', 'w')
    record.write('noise_rate=%s\n' % config.noise_rate)
    record.flush()

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
    tch_net = get_model()
    pretrain_net = get_model()
    test_net = get_model()

    print('| load pretrained net. from checkpoint...')
    checkpoint = torch.load('./checkpoint/%s.pth.tar' % config.checkpoint)
    pretrain_net.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        net.cuda()
        tch_net.cuda()
        pretrain_net.cuda()
        test_net.cuda()
        cudnn.benchmark = True
    pretrain_net.eval()

    for param in tch_net.parameters():
        param.requires_grad = False
    for param in pretrain_net.parameters():
        param.requires_grad = False

    # Instantiate a loss function.
    criterion = torch.nn.CrossEntropyLoss()
    consistent_criterion = torch.nn.KLDivLoss()

    # Instantiate an optimizer to train the model
    optimizer = torch.optim.SGD(
        net.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-3)

    print('\nTraining model')
    print('| Training Epochs = ' + str(config.num_epochs))
    print('| Initial Learning Rate = ' + str(config.lr))
    print('| Optimizer = ' + str(config.optimizer_type))

    init = True
    best_acc = 0
    for epoch in range(1, 1 + config.num_epochs):
        train(epoch)
        # Student validation
        std_results = valid(epoch, net)
        record.write(std_results + '\n')
        print(std_results)
        # Teacher validation
        tch_results = valid(epoch, tch_net)
        record.write(tch_results + '\n')
        record.flush()
        print(tch_results)

    print('\nTesting model')
    checkpoint = torch.load('./checkpoint/%s.pth.tar' % config.id)
    test_net.load_state_dict(checkpoint['state_dict'])
    test()

    record.close()
