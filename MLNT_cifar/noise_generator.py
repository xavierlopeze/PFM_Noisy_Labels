import pandas as pd
import numpy as np
import random

random.seed( 30 )

def generate_noise(r,  symmetric = False, datadir = 'data/', train_file_name = "noisy_label_kv.txt"):

    if symmetric == True:
        #read data of mappings
        df = pd.read_csv(datadir+"/image_set_mapping.csv",usecols = ["directory", "class_number","set"])
        #get only the train data
        df_train = df[df.set == "train"].copy()

        #generate a list of random labels
        set_classes = set(df_train.class_number)
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
                noisy_class.append(row["class_number"])

        df_train["noisy_class"] = noisy_class
        clean_kv = datadir + train_file_name
        # clean_kv = "/content/drive/My Drive/Colab_Notebooks/pfm/data/asdf.txt"
        df_train[["directory", "noisy_class"]].to_csv(clean_kv, sep=' ', index=False,header = False)
        print("\nnoise file " + train_file_name + " generated with noise: " + str(r)+"\n")
        # return(df_train[["directory", "noisy_class"]])



        # clean_kv = datadir + train_file_name
        # clean_kv = "/content/drive/My Drive/Colab_Notebooks/pfm/data/asdf.txt"
        # df_train[["directory", "noisy_class"]].to_csv(clean_kv, sep=' ', index=False,header = False)
