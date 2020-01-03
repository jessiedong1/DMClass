import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import  random as rd
import NLP.GaussianNB as gnb
import NLP.DecisionTree as dt

def main():

    filename = r'D:\Uca\Thesis\NLP\Class_N.csv'
    dataset, X, y = loadCSV(filename)

    Cross_Validation(dataset, 5)

#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    X = dataset.iloc[:,0:13]
    y = dataset.iloc[:,13]
    return dataset, X, y

#Seperate Class by 'Label'
def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))
    #class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

# 5-fold
def Cross_Validation(dataset, n_splits):
    # num_instances = dataset.shape[0]
    # train_set = pd.DataFrame()
    # dataset_copy = dataset.copy()
    class_Y, class_N = Group_Class(dataset)
    num_class_Y = class_Y.shape[0]
    num_class_N = class_N.shape[0]
    print(num_class_Y, num_class_N)
    # print("90+ wines are %d, 90- %d" %(num_class_Y, num_class_N))
    # print(4263/(4263+10086))
    # print(10086/(4263+10086))

    foldsize_Y = int(num_class_Y / n_splits)
    print(foldsize_Y)
    foldsize_N = int(num_class_N / n_splits)
    print(foldsize_N)

    # Split 90+ into 5 folds
    ry = rd.sample(range(0, num_class_Y), num_class_Y)

    # print(ry1)
    ry1 = ry[0:(foldsize_Y)]
    ry2 = ry[foldsize_Y: (foldsize_Y * 2)]
    ry3 = ry[foldsize_Y * 2:(foldsize_Y * 3)]
    ry4 = ry[foldsize_Y * 3:(foldsize_Y * 4)]
    ry5 = ry[foldsize_Y * 4: (num_class_Y)]

    Y_1 = class_Y.iloc[ry1, :]
    Y_2 = class_Y.iloc[ry2, :]
    Y_3 = class_Y.iloc[ry3, :]
    Y_4 = class_Y.iloc[ry4, :]
    Y_5 = class_Y.iloc[ry5, :]
    """
    print(np.sort(ry1))
    print(np.sort(ry2))
    print(np.sort(ry3))
    print(np.sort(ry4))
    print(np.sort(ry5))
    """

    # Split 90- into 5 folds
    rn = rd.sample(range(0, num_class_N), num_class_N)
    # rn = np.array(rn)

    rn1 = rn[0:(foldsize_N)]
    # rn1 = np.sort(rn1)
    # print(rn1)
    rn2 = rn[foldsize_N: (foldsize_N * 2)]
    # rn2 = np.sort(rn2)
    # print(rn2)
    rn3 = rn[foldsize_N * 2:(foldsize_N * 3)]
    rn4 = rn[foldsize_N * 3:(foldsize_N * 4)]
    rn5 = rn[foldsize_N * 4: (num_class_N)]
    # fold_size = int(num_instances / n_splits)

    N_1 = class_N.iloc[rn1, :]
    N_2 = class_N.iloc[rn2, :]
    N_3 = class_N.iloc[rn3, :]
    N_4 = class_N.iloc[rn4, :]
    N_5 = class_N.iloc[rn5, :]

    """
    #print(np.sort(rn1))
    print(np.sort(rn1))
    #print(N_1)
    print(np.sort(rn2))
    #print(N_2)
    print(np.sort(rn3))
    print(np.sort(rn4))
    print(np.sort(rn5))
    """

    # Fold 1
    train_set1 = pd.concat([Y_2, Y_3, Y_4, Y_5, N_2, N_3, N_4, N_5], ignore_index=True)

    # print(train_set1.columns.values[991])
    # print(train_set1.iloc[1,:])
    # print(train_set1.shape)
    test_set1 = pd.concat([Y_1, N_1], ignore_index=True)
    # print(test_set1.shape)
    # print(num_class_Y)
    # print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    # print(num_class_N)
    # print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    # dataset_Y_pro,dataset_N_pro, pro_Ori, matrix, acc, pre, recall
    # print(test_set1.columns)

    # Fold 2
    train_set2 = pd.concat([Y_1, Y_3, Y_4, Y_5, N_1, N_3, N_4, N_5], ignore_index=True)
    # print(train_set1.shape)
    test_set2 = pd.concat([Y_2, N_2], ignore_index=True)

    # print(test_set1.shape)
    # print(num_class_Y)
    # print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    # print(num_class_N)
    # print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])

    # Fold 3
    train_set3 = pd.concat([Y_1, Y_2, Y_4, Y_5, N_1, N_2, N_4, N_5], ignore_index=True)
    # print(train_set1.shape)
    test_set3 = pd.concat([Y_3, N_3], ignore_index=True)

    # print(test_set1.shape)
    # print(num_class_Y)
    # print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    # print(num_class_N)
    # print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])

    # Fold 4
    train_set4 = pd.concat([Y_1, Y_2, Y_3, Y_5, N_1, N_2, N_3, N_5], ignore_index=True)
    # print(train_set1.shape)
    test_set4 = pd.concat([Y_4, N_4], ignore_index=True)

    # print(test_set1.shape)
    # print(num_class_Y)
    # print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    # print(num_class_N)
    # print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])

    # Fold 5
    train_set5 = pd.concat([Y_1, Y_2, Y_3, Y_4, N_1, N_2, N_3, N_4], ignore_index=True)
    # print(train_set1.shape)
    test_set5 = pd.concat([Y_5, N_5], ignore_index=True)


    # print(test_set1.shape)
    # print(num_class_Y)
    # print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    # print(num_class_N)
    # print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])

    # Save data
    # np.save('train_set1', train_set1)
    # np.save('test_set1', test_set1)
    # np.save('train_set2', train_set2)
    # np.save('test_set2', test_set2)
    # np.save('train_set3', train_set3)
    # np.save('test_set3', test_set3)
    # np.save('train_set4', train_set4)
    # np.save('test_set4', test_set4)
    # np.save('train_set5', train_set5)
    # np.save('test_set5', test_set5)

    #Fold 1
    print("Fold1: ")
    acc1,  precision1, recall1 = gnb.NormalDisMean(train_set1,test_set1)
    acc1min, precision1min, recall1min = gnb.NormalDisMin(train_set1, test_set1)


    # Fold 1
    print("Fold2: ")
    acc2, precision2, recall2 = gnb.NormalDisMean(train_set2, test_set2)
    acc2min, precision2min, recall2min = gnb.NormalDisMin(train_set2, test_set2)

    # Fold 1
    print("Fold3: ")
    acc3, precision3, recall3 = gnb.NormalDisMean(train_set3, test_set3)
    acc3min, precision3min, recall3min = gnb.NormalDisMin(train_set3, test_set3)
    # Fold 1
    print("Fold4: ")
    acc4, precision4, recall4 = gnb.NormalDisMean(train_set4, test_set4)
    acc4min, precision4min, recall4min = gnb.NormalDisMin(train_set4, test_set4)

    # Fold 1
    print("Fold5 : ")
    acc5, precision5, recall5 = gnb.NormalDisMean(train_set5, test_set5)
    acc5min, precision5min, recall5min = gnb.NormalDisMin(train_set5, test_set5)


    print("Accuracy mean: {:.4f}".format((acc1+acc2+acc3+acc4+acc5)/5))
    print("Precision mean: {:.4f}".format((precision1 + precision2 + precision3 + precision4 + precision5) / 5))
    print("Recall mean: {:.4f}".format((recall1 + recall2 + recall3 + recall4 + recall5) / 5))

    print("Accuracy min: {:.4f}".format((acc1min + acc2min + acc3min + acc4min + acc5min) / 5))
    print("Precision min: {:.4f}".format((precision1min + precision2min + precision3min + precision4min + precision5min) / 5))
    print("Recall min: {:.4f}".format((recall1min + recall2min + recall3min + recall4min + recall5min) / 5))

    # Fold 1
    print("Fold 1: ")

    train_x1, train_y1 = Split_X_y(train_set1)
    test_x1, test_y1 = Split_X_y(test_set1)
    acc1, precision1, recall1 = dt.NaiveB(train_x1, train_y1, test_x1, test_y1)
    # Fold 2
    print("Fold 2: ")
    train_x2, train_y2 = Split_X_y(train_set2)
    test_x2, test_y2 = Split_X_y(test_set2)
    acc2, precision2, recall2 = dt.NaiveB(train_x2, train_y2, test_x2, test_y2)

    # Fold 3
    print("Fold 3: ")
    train_x3, train_y3 = Split_X_y(train_set3)
    test_x3, test_y3 = Split_X_y(test_set3)
    acc3, precision3, recall3 = dt.NaiveB(train_x3, train_y3, test_x3, test_y3)

    # Fold 4
    print("Fold 4: ")
    train_x4, train_y4 = Split_X_y(train_set4)
    test_x4, test_y4 = Split_X_y(test_set4)
    acc4, precision4, recall4 = dt.NaiveB(train_x4, train_y4, test_x4, test_y4)

    # Fold 5
    print("Fold 5: ")
    train_x5, train_y5 = Split_X_y(train_set5)
    test_x5, test_y5 = Split_X_y(test_set5)
    acc5, precision5, recall5 = dt.NaiveB(train_x5, train_y5, test_x5, test_y5)
    print("Accuracy mean: {:.4f}".format((acc1+acc2+acc3+acc4+acc5)/5))
    print("Precision mean: {:.4f}".format((precision1 + precision2 + precision3 + precision4 + precision5) / 5))
    print("Recall mean: {:.4f}".format((recall1 + recall2 + recall3 + recall4 + recall5) / 5))








def Split_X_y(dataset):
    X = dataset.iloc[:, 0:13]
    y = dataset.iloc[:, 13]

    return X, y

main()