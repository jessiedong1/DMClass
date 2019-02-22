import pandas as pd
import random as rd
import math

def main():
    filename ='D:\Spring2019\DataMining\Dataset\Output.csv'
    rd.seed(1)
    #print(filename)

    dataset = loadCSV(filename)
    #print(dataset.iat[0,0])
    atts = dataset.columns.values
    atts = atts[8:((len(atts))-1)]



    #print(dataset['GRAPEFRUIT PEEL'])
    #print(atts)
    #print(dataset.values)
    #fold = Cross_Validation(dataset, 4)
    dataset_Y, dataset_N = Group_Class(dataset)
    num_Y = dataset_Y.shape[0]
    num_X = dataset_N.shape[0]
    #print(num_X,num_Y)
   # print(dataset_Y)
    #dataset_Y = pd.DataFrame(dataset_Y)
    #dataset_N = pd.DataFrame(dataset_N)
    #print(dataset_Y)
    #att = 'GRAPEFRUIT'
    #pro0,pro1 =Cal_Pro(dataset_Y,att)
   # print(pro0, ' ', pro1)

    dataset_Y_pro = Cal_All_Attributs(dataset_Y,atts)
    dataset_N_pro = Cal_All_Attributs(dataset_N,atts)
    #print(dataset_Y_pro)
    print(dataset_N_pro)


"""
    array = {'name':[1,2,4],
             'age':[20,30,40]}
    labels = ['1','2','3']
    df = pd.DataFrame(array,labels)
"""

#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    return dataset

#K-fold
def Cross_Validation(dataset, n_splits):
    num_instances = dataset.shape[0]
    train_set = pd.DataFrame()
    dataset_copy = dataset.copy()
    fold_size = int(num_instances/n_splits)
    for i in range(n_splits):

        index = rd.randrange(num_instances)

    return [train_set, dataset_copy]

#Seperate Class by 'Label'
def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))
    #class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

#Calculate the probablity of one attributs in one class
def Cal_Pro(class_Y,att):
    #print(class_Y)
    #att_0, att_1= (g for _, g in class_Y.groupby(att))
    #print(att_0, att_1)
    data = class_Y[att]
    att_0 =0
    att_1 =0
    for i in range((class_Y.shape[0])):
        if data.iat[i] == 0:
            att_0 += 1
        else:
            att_1 += 1
    pro_0 = float(att_0/(len(class_Y)))
    pro_1 = float(att_1/(len(class_Y)))
    return [pro_0,pro_1]

#Calculate the probabblity of all the attributes in the same class
#Can only calculate one class each time
def Cal_All_Attributs(class_Y, atts):
    array = {atts[i]: [Cal_Pro(class_Y,(atts[i]))]
                 for i in range(len(atts))}
    labels = ['ATT=0', 'ATT=1']
    ar = pd.DataFrame(array,labels)
    return ar
#Based on training set, calculate the bayesian
def Get_Pro(dataset,dataset_Y_Pro):

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if(dataset.iat[i,j]==0):
                print()








main()
