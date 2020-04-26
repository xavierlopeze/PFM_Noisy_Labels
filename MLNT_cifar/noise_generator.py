import pandas as pd
import numpy as np
import random

random.seed( 30 )

def generate_noise(r,  symmetric = False, datadir = 'data/', train_file_name = "noisy_label_kv.txt", noise_validation = False,test_file_name = "noisy_validation.txt"):

    if symmetric == True:

        #GENERATE NOISE ON TRAIN DATASET:

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


        #GENERATE NOISE ON VALIDATION DATASET:
        if noise_validation == True:
            df_data_mapping = pd.read_csv(datadir+'/image_set_mapping.csv')
            df_clean_labels = pd.read_csv(datadir+'/clean_label_kv.txt', sep = " ", header = None)
            df_clean_labels.columns = ['directory', "label"]

            #get a full dataframe containing sets and classes
            df = df_data_mapping.merge(df_clean_labels, how = 'left', right_on = "directory", left_on = "directory",  left_index = False, right_index = False)

            #get separate dataframes for validation and test
            val_df  = df[df.set == "validation"][["directory","class_number","set","label"]].copy()
            test_df = df[df.set == "test"][["directory","class_number","set","label"]].copy()

            #generate a list of random labels
            set_classes = set(df.label)
            randomlist = []
            for i in range(0,val_df.shape[0]):
                n = random.randint(min(set_classes),max(set_classes))
                randomlist.append(n)
            val_df["random_class"] = randomlist

            #column of probabilities (uniform)
            val_df["p"] = np.random.uniform(0,1,val_df.shape[0])

            #use the random list to generate noise in effective percentage r
            N_CLASSES = len(set_classes)
            NOISE_PERCENTAGE = r*N_CLASSES/(N_CLASSES-1)
            noisy_class = []
            for index, row in val_df.iterrows():
                if row["p"]<= NOISE_PERCENTAGE:
                    noisy_class.append( row["random_class"])
                else:
                    noisy_class.append(row["class_number"])

            val_df["noisy_class"] = noisy_class

            #Set label to noisy and select dataframe to be exported
            val_df["class_number"] = val_df["noisy_class"]
            df_export = val_df[["directory","class_number"]].append(test_df)[["directory", "class_number"]]

            #generate file with noisy labels on test
            df_export.to_csv(clean_kv, sep=' ', index=False,header = False)
            print("\nnoise file " + test_file_name + " generated with noise: " + str(r)+"\n")
