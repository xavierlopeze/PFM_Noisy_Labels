import pandas as pd
import numpy as np
import random

# Most frequently modified parameters
# -------------------------------------------- #
#
# noise_validation = True

# Record parameters to wandb
use_wandb = True
noise_generator =  True
noise_validation = True
r = 0 #noise noise_rate
generate_logs = True
batch_size = 64
start_iter = 500 #500
reps = True

# if symmetric == True:
#     str_sym = "sim"
# else:
#     str_sym = 'asym'
#
# if noise_validation == True:
#     str_noiseval = "_noise_validation"
# else:
#     str_noiseval = ""

checkpoint = 'cross_entropy'
id = 'MLNT'

if noise_generator == True:
    checkpoint = checkpoint + '_sim_noise_' + str(int(100*r))
    id = id                 + '_sim_noise_' + str(int(100*r))
    
if noise_validation == True:
    checkpoint  += '_noise_val'
    id          += '_noise_val'

if reps == True:
    id += 'reps'



# Hyper-params. (baseline)
# -------------------------------------------- #
num_workers = 10
start_epoch = 1
num_epochs = 5
shuffle = True
pretrained = True
seed = 7
lr =  0.008
optimizer_type = 'SGD'
momentum = 0.9
weight_decay = 0.001

# Hyper-params. (main)
# -------------------------------------------- #
meta_lr = 0.0008  # meta learning_rate
num_fast = 10  # number of random perturbations
perturb_ratio = 0.5  # ratio of random perturbations

mid_iter = 2000
alpha = 0.02
eps = 0.99  # Running average of model weights
gpuid = 1
param_epoch = 20

# Image pre-processing
# -------------------------------------------- #
image_size = 256
crop = 32

# -------------------------------------------- #

# drive_dir = '.'
# data_dir = './data/'
drive_dir = '/content/drive/My Drive/Colab_Notebooks/git/PFM_Noisy_Labels/FOOD101'
data_dir =  '/content/drive/My Drive/Colab_Notebooks/git/PFM_Noisy_Labels/FOOD101/data/'#'/data/'

#----------------------------------------------#
#Dataloader files
train_dir = 'clean_train_key_list.txt' #directories file for train, used in discarding

train_file = 'clean_label_kv.txt'
valid_test_file = 'clean_label_kv.txt'

# Choose a network
# -------------------------------------------- #
# model = "resnet18"
# model = "resnet34"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
# model = resnext50_32x4d
# model = resnext101_32x8d
# model = wide_resnet50_2
# model = wide_resnet101_2







#Label FILES
train_labels_file = train_file
test_validation_labels_file = 'clean_label_kv.txt'


#GENERATE NOISE FILES
# generate_file = True
#
# import noise_generator
# N_CLASSES = 10 #Depends on the dataset (10 for cifar)s
# train_file_name = 'noisy_label_kv' +str(int(noise_rate*100))+'_' + str_sym +'.txt'
# test_file_name =  'noisy_label_validation' +str(int(noise_rate*100)) +'_' + str_sym +'.txt'
#
# if noise_validation == True:
#     test_validation_labels_file = test_file_name
#
# if generate_file == True:
#     noise_generator.generate_noise(noise_rate,symmetric, data_dir, train_file_name, noise_validation, test_file_name)
#


#NOISE GENERATOR
if noise_generator == True:

    random.seed(0)
    np.random.seed(0)

    df_labels = pd.read_csv(drive_dir+"/data/clean_label_kv.txt", header = None, sep = " ")
    df_labels.columns = ["dir", "clean_label"]

    df_train = pd.read_csv(drive_dir+"/data/clean_train_key_list.txt", header = None, sep = " ")
    df_train.columns = ["dir"]

    df_train = df_train.merge(df_labels, how = 'left', left_on = "dir", right_on = "dir")

    #generate a list of random labels
    set_classes = set(df_train.clean_label)
    randomlist = []
    for i in range(0,df_train.shape[0]):
        n = random.randint(min(set_classes),max(set_classes))
        randomlist.append(n)
    df_train["random_class"] = randomlist

    #column of probabilities (uniform)
    df_train["p"] = np.random.uniform(0,1,df_train.shape[0])

    #use the random list to generate noise in effective percentage r
    N_CLASSES = len(set_classes)
    NOISE_PERCENTAGE = r*N_CLASSES/(N_CLASSES-1)
    noisy_class = []
    for index, row in df_train.iterrows():
        if row["p"]<= NOISE_PERCENTAGE:
            noisy_class.append( row["random_class"])
        else:
            noisy_class.append(row["clean_label"])

    df_train["noisy_class"] = noisy_class

    df_labels["noisy_class"] = df_labels["clean_label"]
    df_train_relabeled = df_train[["dir","noisy_class"]].copy()
    # df_full = df_labels[["dir","noisy_class"]].copy()
    # df_full.update(df_train_relabeled)
    df_full = df_train_relabeled

    filename = drive_dir + '/data/' + 'noisy_label_kv'+str(int(r*100))+'_sim.txt'
    df_full["noisy_class"] = [int(x) for x in df_full["noisy_class"]]
    df_full[["dir", "noisy_class"]].to_csv(filename, sep=' ', index=False,header = False)
    print("\nnoise file " + filename + " generated with noise: " + str(r)+"\n")
    # return(df_train[["directory", "noisy_class"]])

    train_labels_file = 'noisy_label_kv'+str(int(r*100))+'_sim.txt'

    # if noise_validation == True:
    #     df_labels = pd.read_csv(drive_dir+"/data/" + train_labels_file, header = None, sep = " ")
    #     df_labels.columns = ["dir", "clean_label"]
    
    #     df_val = pd.read_csv(drive_dir+"/data/clean_val_key_list.txt", header = None, sep = " ")
    #     df_val.columns = ["dir"]
    
    #     df_val = df_val.merge(df_labels, how = 'left', left_on = "dir", right_on = "dir")


    #     #generate a list of random labels
    #     randomlist = []
    #     for i in range(0,df_val.shape[0]):
    #         n = random.randint(min(set_classes),max(set_classes))
    #         randomlist.append(n)
    #     df_val["random_class"] = randomlist
    
    #     #column of probabilities (uniform)
    #     df_val["p"] = np.random.uniform(0,1,df_val.shape[0])
    
    #     #use the random list to generate noise in effective percentage r
    #     N_CLASSES = len(set_classes)
    #     NOISE_PERCENTAGE = r*N_CLASSES/(N_CLASSES-1)
    #     noisy_class = []
    #     for index, row in df_val.iterrows():
    #         if row["p"]<= NOISE_PERCENTAGE:
    #             noisy_class.append( row["random_class"])
    #         else:
    #             noisy_class.append(row["clean_label"])
    
    #     df_val["noisy_class"] = noisy_class
    
    #     df_labels["noisy_class"] = df_labels["clean_label"]
    #     df_val_relabeled = df_val[["dir","noisy_class"]].copy()
    #     df_full = df_labels[["dir","noisy_class"]].copy()
    #     df_full.update(df_val_relabeled, overwrite=True)
    
    #     filename = drive_dir + '/data/' + 'noisy_label_kv'+str(int(r*100))+'_sim.txt'
    #     df_full["noisy_class"] = [int(x) for x in df_full["noisy_class"]]
    #     df_full[["dir", "noisy_class"]].to_csv(filename, sep=' ', index=False,header = False)
    #     print("\nnoise file " + filename + " generated with noise: " + str(r)+"\n")
    #     # return(df_train[["directory", "noisy_class"]])
       
    #     test_validation_labels_file  = 'noisy_label_kv'+str(int(r*100))+'_sim.txt'



if use_wandb== True:
    import wandb

    project_name = "Food101_Noise_Server"

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
