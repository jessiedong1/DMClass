import pandas as pd
import random as rd
import numpy as np
import math

#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    dataset['Label'] = np.where(dataset['Score'] < 90, 0, 1)
    return dataset

#Seperate Class by 'Label'
def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))
    #class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

#Calculate the probablity of one attributs in one class binary values
def Cal_Pro_att_Ori(class_Y,att,pro_Y):
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

#No estimate
    #pro_0 = float(att_0/(len(class_Y)))
    #pro_1 = float(att_1/(len(class_Y)))
#Laplace estimate
    pro_0 = float((att_0 + 1) / (len(class_Y) + 2))
    pro_1 = float((att_1 + 1) / (len(class_Y) + 2))
#M- estimate
    #pro_0 = float((att_0 + 30*pro_Y) / (len(class_Y) + 30))
    #pro_1 = float((att_1 + 30*pro_Y) / (len(class_Y) + 30))
    return pro_0, pro_1
def Cal_All_Attributs_Ori(class_Y, atts,pro_Y,):
    size = 984
    pro_0 = [0]*size
    pro_1 = [0]*size
    for i in range(size):
        pro_0[i], pro_1[i] = Cal_Pro_att_Ori(class_Y,atts[i],pro_Y)
        #pro_1[i] = Cal_Pro_att1(class_Y,atts[i])
    array = { 'ATT=0' : pro_0,
                'ATT=1' : pro_1}
    ar = pd.DataFrame(array, atts)
    return ar

#Handle zero frequency in continous values
def Zero_Frequency_min(dataset):
    for i in range(dataset.shape[0]-1):
        if dataset.iloc[i] == 0:
            dataset.iloc[i] = dataset[dataset > .01].min()

    return dataset

#Based on training set, calculate the bayesian
def Get_Pro(test_data,dataset_Y_Pro, class_Y):
    atts = test_data.columns[4:1001]
    test_data = test_data[atts]
    #print(test_data.iloc[0,0])
    ROWS = test_data.shape[0]
    COLS = 997
    pro = [[0]* COLS]*ROWS
    pro = pd.DataFrame(pro, columns=atts,dtype=float)
    for i in range(test_data.shape[0]):
        for j in range(0,984):

            if(test_data.iloc[i][j] == 0):
                pro.iloc[i][j] = dataset_Y_Pro.iloc[j][0]
            else:
                pro.iloc[i][j] = dataset_Y_Pro.iloc[j][1]
    #ar = pd.DataFrame(pro, columns=atts)
    #print(ar)
    class_Y = class_Y.iloc[:, 988:1001]
    train_set_mean = class_Y.mean()
    train_set_var = class_Y.var()

    train_set_mean = Zero_Frequency_min(train_set_mean)
    train_set_var = Zero_Frequency_min(train_set_var)

    for i in range(test_data.shape[0]):
        for j in range(984, 997):
            pro.iloc[i][j] = (1 / (math.sqrt(2 * math.pi * train_set_var[j-984]))) * (
                math.exp(-((test_data.iloc[i][j-984] - train_set_mean[j-984]) ** 2) / (2 * train_set_var[j-984])))

    return pro

def Cal_NB(test_pro):
    pro = test_pro.product(axis = 1)
    return pro

def Classify_NB(test_pro_Y, test_pro_N, pro_Y, pro_N):
    # Calculate NB in Y
    pros_Y = Cal_NB(test_pro_Y)
    pros_Y = pro_Y * pros_Y
    pro_Ori = pd.DataFrame(pros_Y.values, columns=['Probability_Y'])
    pros_N = Cal_NB(test_pro_N)
    pros_N = pro_N * pros_N
    pro_Ori['Probability_N'] = pros_N.values

    # Classify the test dataset
    test_size = pros_Y.shape[0]
    sample_class = [99] * test_size
    for i in range(test_size):
        if (pro_Ori.iloc[i, 0] >= pro_Ori.iloc[i, 1]):
            sample_class[i] = 1
        else:
            sample_class[i] = 0
    pro_Ori['Predict_Class'] = sample_class



    return pro_Ori

def Confusion_Matrix(pro_Ori):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(pro_Ori.shape[0]):
        if((pro_Ori.iloc[i, 2] == 1)and (pro_Ori.iloc[i,3] == 1)):
            tp = tp+1
        elif((pro_Ori.iloc[i, 2] == 0)and (pro_Ori.iloc[i,3] == 1)):
            fn = fn+1
        elif ((pro_Ori.iloc[i, 2] == 1) and (pro_Ori.iloc[i, 3] == 0)):
            fp = fp+1
        else:
            tn = tn+1

    array = {'Predicted Class = 90+' :[tp,fp],
             'Predicted Class = 90-' : [fn,tn]}

    actual_Class = ['Actual Class = 90+ ', 'Actual Class = 90-']
#Put two list into a dataframe
    matrix = pd.DataFrame(array,actual_Class)
    #matrix.to_csv(r'D:\Spring2019\DataMining\Output\Con_Matrix.csv')
    print("Confusion Matrix: ")
    print(matrix)
    precision = 0
    recall = 0
    acc = float((tp+tn)/(tp+fn+fp+tn))
    print("Accuracy: {:.4f}".format(acc))
    if(tp != 0):
        precision = float(tp/(tp+fp))
        recall = float(tp/(tp+fn))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
    print()

    return acc, precision, recall
#Print accuracy, recall and precision

#Get the result
def NBRresult(train_set, test_set):
    rd.seed(1)
    # print(filename)


    # print(dataset.iat[0,0])
    # Extract the attributes to be calculated
    atts = train_set.columns
    #print(atts)
    atts = atts[4:988]

    Total_sample = train_set.shape[0]
    train_set_Y, train_set_N = Group_Class(train_set)

    num_Y = train_set_Y.shape[0]
    num_N = train_set_N.shape[0]
    # Calculate class distrubution
    pro_Y = float(num_Y / Total_sample)
    pro_N = float(num_N / Total_sample)
    print(pro_Y,pro_N)

    #return matrix, acc, precision,recall

    # print(len(atts))
    # data = dataset[atts]
    # print(data)
    # print(dataset.iat[0,0])

    # print(dataset['GRAPEFRUIT PEEL'])
    # print(atts)
    # print(dataset.values)
    # fold = Cross_Validation(dataset, 4)

    # Cross_Validation(dataset, 5)

    # print(num_X,num_Y)

    # dataset_Y = pd.DataFrame(dataset_Y)
    # dataset_N = pd.DataFrame(dataset_N)
    # print(dataset_Y)
    # Testing Cal_Pro_att0 function
    #att = 'GRAPEFRUIT'
    #pro0, pro1 =Cal_Pro_att_Ori(train_set_Y,att)
    #print(pro0, ' ', pro1)

    # Calculating all the attributes probabliaty in Y/N class without any fix
    dataset_Y_pro = Cal_All_Attributs_Ori(train_set_Y, atts,pro_Y)
    dataset_N_pro = Cal_All_Attributs_Ori(train_set_N, atts,pro_N)
    #dataset_Y_pro.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro.csv')
    #dataset_N_pro.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro.csv')
    #print(dataset_Y_pro)
    #print(dataset_N_pro)

    # Initilize test dataset
    #test_data = dataset.iloc[50:250, :]
    test_data_Labels = test_set['Label']

    # Get the NB in Y class
    test_pro_Y = Get_Pro(test_set, dataset_Y_pro, train_set_Y)

    # Get the NB in N class
    test_pro_N = Get_Pro(test_set, dataset_N_pro, train_set_N)

    # print(test_pro)

    # Calculate NB
    pro_Ori = Classify_NB(test_pro_Y, test_pro_N, pro_Y, pro_N)
    pro_Ori['Actual_Labels'] = test_data_Labels.values
    #pro_Ori.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_ALL_Pro.csv')
    #Print the Final probablity
    #print(pro_Ori)
    acc, pre, recall = Confusion_Matrix(pro_Ori)

    return dataset_Y_pro,dataset_N_pro, pro_Ori, acc, pre, recall

#
# def main():
#
#
#
#     filename = r'D:\Uca\Thesis\NLP\Wine1855.csv'
#     # Load the data
#     dataset = loadCSV(filename)
#     train_set = dataset.iloc[200:300,:]
#
#     test_data = dataset.iloc[100:159,:]
#     atts = train_set.columns
#     # print(atts)
#     atts = atts[4:988]
#     test_data = test_data[atts]
#
#     atts = train_set.columns
#     # print(atts)
#     atts = atts[4:988]
#
#     Total_sample = train_set.shape[0]
#     train_set_Y, train_set_N = Group_Class(train_set)
#
#     num_Y = train_set_Y.shape[0]
#     num_N = train_set_N.shape[0]
#     # Calculate class distrubution
#     pro_Y = float(num_Y / Total_sample)
#     pro_N = float(num_N / Total_sample)
#     print(pro_Y, pro_N)
#     dataset_Y_pro = Cal_All_Attributs_Ori(train_set_Y, atts, pro_Y)
#     dataset_N_pro = Cal_All_Attributs_Ori(train_set_N, atts, pro_N)
#     # print(test_data.iloc[0,0])
#     ROWS = test_data.shape[0]
#     COLS = test_data.shape[1]
#     pro = [[0] * COLS] * ROWS
#     pro = pd.DataFrame(pro, columns=atts, dtype=float)
#     for i in range(test_data.shape[0]):
#         for j in range(0, 984):
#             if (test_data.iloc[i][j] == 0):
#                 pro.iloc[i][j] = dataset_Y_pro.iloc[j][0]
#             else:
#                 pro.iloc[i][j] = dataset_Y_pro.iloc[j][1]
#     # ar = pd.DataFrame(pro, columns=atts)
#     # print(ar)
#     print(pro)
#
#
#
# main()