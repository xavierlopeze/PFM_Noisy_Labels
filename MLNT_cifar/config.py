# Hyper-params. (baseline)
# -------------------------------------------- #
num_workers = 10
start_epoch = 80
num_epochs = 120
batch_size = 128
shuffle = True
pretrained = False
seed = 7
lr = 0.2
optimizer_type = 'SGD'
checkpoint = 'cross_entropy_r40_ResNet34_BasicBlock_assym'
noise_rate = 0.4
momentum = 0.9
weight_decay = 0.0001

# Hyper-params. (main)
# -------------------------------------------- #
id = 'MLNT_r40_ResNet34_BasicBlock_assym'
meta_lr = 0.4  # meta learning_rate
num_fast = 10  # number of random perturbations
perturb_ratio = 0.5  # ratio of random perturbations
start_iter = 500
mid_iter = 2000
alpha = 0.2
eps = 0.99  # Running average of model weights
gpuid = 1
param_epoch = 20

# Image pre-processing
# -------------------------------------------- #
image_size = 32
crop = 4

# -------------------------------------------- #

#drive_dir = '.'
#data_dir = './data/'
drive_dir = '/content/drive/My Drive/Colab_Notebooks/git/PFM_Noisy_Labels/MLNT_cifar'
data_dir =  '/content/drive/My Drive/Colab_Notebooks/git/PFM_Noisy_Labels/MLNT_cifar/data/'#'/data/'

#----------------------------------------------#
#Dataloader files
train_dir = 'clean_train_key_list.txt' #directories file for train, used in discarding

train_file = 'noisy_label_kv' +str(noise_rate*100)+ '.txt'
valid_test_file = 'clean_label_kv.txt'

# Choose a network
# -------------------------------------------- #
# model = "resnet18"
# model = "resnet34"
model = "resnet34"
# model = "resnet101"
# model = "resnet152"
# model = resnext50_32x4d
# model = resnext101_32x8d
# model = wide_resnet50_2
# model = wide_resnet101_2

# Record parameters to wandb
use_wandb = True


#Label FILES
train_labels_file = 'noisy_label_kv' +str(int(noise_rate)*100)+ '.txt'
test_validation_labels_file = 'clean_label_kv.txt'


#GENERATE NOISE FILES
generate_file = True
noise_validation = False


import noise_generator
N_CLASSES = 10 #Depends on the dataset (10 for cifar)s
symmetric = False
train_file_name = 'noisy_label_kv' +str(int(noise_rate)*100)+ '.txt'
test_file_name =  'noisy_label_validation' +str(int(noise_rate)*100)+ '.txt'

if noise_validation == True:
    test_validation_labels_file = test_file_name

if generate_file == True:
    noise_generator.generate_noise(noise_rate,symmetric, data_dir, train_file_name, noise_validation, test_file_name)


if use_wandb== True:
    import wandb

    project_name = "Cifar_Experiment"

    args = {"num_workers": num_workers,
            "start_epoch":start_epoch,
            "num_epochs":num_epochs,
            "batch_size":batch_size,
            "shuffle":shuffle,
            "pretrained":pretrained,
            "seed":seed,
            "lr":lr,
            "optimizer_type":optimizer_type,
            "checkpoint":checkpoint,
            "noise_rate":noise_rate,
            "id":id,
            "meta_lr":meta_lr,
            "num_fast":num_fast,
            "perturb_ratio":perturb_ratio,
            "start_iter":start_iter,
            "mid_iter":mid_iter,
            "alpha":alpha,
            "eps":eps,
            "gpuid":gpuid,
            "image_size":image_size,
            "crop":crop,
            "data_dir":data_dir}

    wandb.init(project=project_name, config=args)
