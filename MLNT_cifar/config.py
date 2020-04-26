# Hyper-params. (baseline)
# -------------------------------------------- #
num_workers = 10
start_epoch = 1
num_epochs = 100
batch_size = 30
shuffle = True
pretrained = True
seed = 7
lr = 0.0008
optimizer_type = 'SGD'
checkpoint = 'cross_entropy_r10_dataloader_defaultparam'
noise_rate = 0.1

# Hyper-params. (main)
# -------------------------------------------- #
id = 'MLNT_r10_dataloader_defaultparam'
meta_lr = 0.02  # meta learning_rate
num_fast = 10  # number of random perturbations
perturb_ratio = 0.5  # ratio of random perturbations
start_iter = 500
mid_iter = 2000
alpha = 1
eps = 0.99  # Running average of model weights
gpuid = 1

# Image pre-processing
# -------------------------------------------- #
image_size = 256
crop = 0

# -------------------------------------------- #

#drive_dir = '.'
#data_dir = './data/'
drive_dir = '/content/drive/My Drive/Colab_Notebooks/pfm' #si es llenà¸£à¸‡a en local deixar en blanc ""
data_dir = '/content/drive/My Drive/Colab_Notebooks/pfm/data/'

#----------------------------------------------#
#Dataloader files
train_file = 'noisy_label_kv.txt'
valid_test_file = 'clean_label_kv.txt'

# Choose a network
# -------------------------------------------- #
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
# model = resnext50_32x4d
# model = resnext101_32x8d
# model = wide_resnet50_2
# model = wide_resnet101_2

# Record parameters to wandb
use_wandb = True

#GENERATE NOISE FILES
generate_file = True
noise_validation = True


import noise_generator
N_CLASSES = 10 #Depends on the dataset (10 for cifar)s
symmetric = True
train_file_name = 'noisy_label_kv.txt'
test_file_name =  'test_with_noisy_label_validation_kv.txt'

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
