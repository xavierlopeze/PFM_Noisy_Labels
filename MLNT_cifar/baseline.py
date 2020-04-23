# Internal files
import config
import dataloader
import models
import models32

# Pytorch libraries
import torch
import torch.backends.cudnn as cudnn

# Other lib.
import os
import time
import wandb


def get_model():
    # Get model from config
    # image size: 32x32
    # -------------------------------------------- #
    if config.model == "resnet20":
        model = models32.resnet20()
    elif config.model == "resnet32":
        model = models32.resnet32()
    elif config.model == "resnet44":
        model = models32.resnet44()
    elif config.model == "resnet56":
        model = models32.resnet56()
    elif config.model == "resnet110":
        model = models32.resnet110()
    elif config.model == "resnet1202":
        model = models32.resnet1202()
    # image size: 224x224
    # -------------------------------------------- #
    elif config.model == "resnet18":
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
    model.fc = torch.nn.Linear(in_features=model.fc.in_features,
                               out_features=config.out_features)
    return model


def scheduler(epoch: int):
    global lr
    lr = config.lr
    if epoch > config.start_epoch:
        lr = lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, config.drive_dir + filename)


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    scheduler(epoch)
    for step, (inputs, targets) in enumerate(train_loader):
        init_time = time.time()
        if use_cuda:  # GPU settings
            (inputs, targets) = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # Grab training results
        print("| Epoch: {}/{}, step: {}/{}, loss: {:.3f}, acc: {:.3f},"
              " time: {:.3f}".format(epoch,
                                     config.num_epochs,
                                     step + 1,
                                     len(train_loader.dataset) // config.batch_size + 1,
                                     loss.data.item(),
                                     float(correct) / float(total),
                                     time.time() - init_time),
              end="\r")


def valid(epoch):
    global best_acc
    net.eval()
    # valid_loss = 0
    correct = 0
    total = 0
    for step, (inputs, targets) in enumerate(valid_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        # valid_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Grab validation results
    valid_acc = float(correct) / float(total)
    valid_results = ("| Epoch: {}/{}, val_loss: {:.3f}, val_acc: {:.3f}, "
                     "lr: {:.6f}".format(epoch,
                                         config.num_epochs,
                                         loss.data.item(),
                                         valid_acc,
                                         lr))
    record.write(valid_results + '\n')
    record.flush()

    print(valid_results)
    # wandb.log({'epoch': epoch, 'accy_val': valid_results})

    # Save checkpoint when best model
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('| Saving Best Model ...', end="\r")
        save_checkpoint(
            state={'state_dict': net.state_dict(), },
            filename='/checkpoint/%s_r%d.pth.tar' % (
                config.baseline_id, 100 * config.r))


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
    test_acc = float(correct) / float(total)
    test_results = "| test_loss: {:.3f}, test_acc: {:.3f}".format(
        loss.data.item(), test_acc)
    record.write(test_results)
    record.flush()

    print(test_results)
    # wandb.log({'test_acc': test_acc})


if __name__ == '__main__':

    # Checkpoint dir.
    # os.mkdir('checkpoint')
    record = open(config.drive_dir + '/checkpoint/%s_r%d.txt' % (
        config.baseline_id, 100 * config.r), 'w')
    record.flush()

    # Get the original_dataset
    loader = dataloader.KeyDataLoader()
    train_loader, valid_loader, test_loader = loader.run()

    # Hyper Parameter settings
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    use_cuda = torch.cuda.is_available()

    # Networks setup
    print('\nModel setup')
    print('| Building network: {}'.format(config.model))
    # net = get_model()
    # test_net = get_model()
    net = models32.ResNet18()
    test_net = models32.ResNet18()

    if use_cuda:
        net.cuda()
        test_net.cuda()
        cudnn.benchmark = True

    # Instantiate a loss function.
    criterion = torch.nn.CrossEntropyLoss()

    # Instantiate an optimizer to train the model.
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    print('\nTraining model')
    print('| Training Epochs = ' + str(config.num_epochs))
    print('| Initial Learning Rate = ' + str(config.lr))
    print('| Optimizer = ' + str(config.optimizer_type))

    best_acc = 0
    for epoch in range(1, 1 + config.num_epochs):
        train(epoch)
        valid(epoch)

    print('\nTesting model')
    checkpoint = torch.load(
        config.drive_dir + '/checkpoint/%s_r%d.pth.tar' % (
            config.baseline_id, 100 * config.r))
    test_net.load_state_dict(checkpoint['state_dict'])
    test()




