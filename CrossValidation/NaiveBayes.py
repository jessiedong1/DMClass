import pandas as pd
import random as rd
import NavieBayes.NBClassifier as nb
import numpy as np
def main():
    filename = 'D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv'
    # Load the data
    dataset = nb.loadCSV(filename)
    trainset1 = np.load('train_set1.npy')
    #print(trainset1)
    trainset2 = np.load('train_set2.npy')
    trainset3 = np.load('train_set3.npy')
    trainset4 = np.load('train_set4.npy')
    trainset5 = np.load('train_set5.npy')

    testset1 = np.load('test_set1.npy')
    testset2 = np.load('test_set2.npy')
    testset3 = np.load('test_set3.npy')
    testset4 = np.load('test_set4.npy')
    testset5 = np.load('test_set5.npy')

    print("Result for fold 1")
    dataset_Y_pro_1, dataset_N_pro_1, Pro_fold_1, acc1, precision1, recall1 = nb.NBRresult(trainset1, testset1)
    dataset_Y_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_1.csv')
    dataset_N_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_1.csv')
    Pro_fold_1.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_1.csv')

    print("Result for fold 2")
    dataset_Y_pro_2, dataset_N_pro_2, Pro_fold_2, acc2, precision2, recall2 = nb.NBRresult(trainset2, testset2)
    dataset_Y_pro_2.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_2.csv')
    dataset_N_pro_2.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_2.csv')
    Pro_fold_2.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_2.csv')


    print("Result for fold 3")
    dataset_Y_pro_3, dataset_N_pro_3, Pro_fold_3,acc3, precision3, recall3 = nb.NBRresult(trainset3,testset3)
    dataset_Y_pro_3.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_3.csv')
    dataset_N_pro_3.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_3.csv')
    Pro_fold_3.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_3.csv')


    print("Result for fold 4")
    dataset_Y_pro_4, dataset_N_pro_4, Pro_fold_4,acc4, precision4, recall4 =nb.NBRresult(trainset4,testset4)
    dataset_Y_pro_4.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_4.csv')
    dataset_N_pro_4.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_4.csv')
    Pro_fold_4.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_4.csv')


    # Fold 5

    print("Result for fold 5")
    dataset_Y_pro_5, dataset_N_pro_5, Pro_fold_5,acc5, precision5, recall5 = nb.NBRresult(trainset5,testset5)
    dataset_Y_pro_5.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_5.csv')
    dataset_N_pro_5.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_5.csv')
    Pro_fold_5.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_5 .csv')


    # Average of acc, precision, recall

    acc_avg = float((acc1+acc2+acc3+acc4+acc5)/5)
    precision_avg = float((precision1+precision2+precision3+precision4+precision5)/5)
    recall_avg = float((recall1+recall2+recall3+recall4+recall5)/5)
    print("Accuracy Ave: {}".format(acc_avg))
    print("Precision Ave: {}".format(precision_avg))
    print("Recall Ave: {}".format(recall_avg))



main()