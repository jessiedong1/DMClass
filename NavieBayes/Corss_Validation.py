import pandas as pd
import random as rd
import NavieBayes.NBClassifier as nb
import numpy as np

def main():
    rd.seed(0)
    filename = 'D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv'
    # Load the data
    dataset = nb.loadCSV(filename)

    #r1= rd.sample(range(0, dataset.shape[0]), 200)
    #r2 = rd.sample(range(0, dataset.shape[0]), 200)

    #print(r1)
    #print(r2)
    #print(dataset.shape)
    Cross_Validation(dataset, 5)

# 5-fold
def Cross_Validation(dataset, n_splits):
    #num_instances = dataset.shape[0]
    #train_set = pd.DataFrame()
    #dataset_copy = dataset.copy()
    class_Y, class_N = nb.Group_Class(dataset)
    num_class_Y = class_Y.shape[0]
    num_class_N = class_N.shape[0]
    print(num_class_Y, num_class_N)

    foldsize_Y = int(num_class_Y / n_splits)
    foldsize_N = int(num_class_N / n_splits)

#Split Class Y into 5 sets
    ry1 = rd.sample(range(0, num_class_Y), foldsize_Y)
    #print(ry1)
    ry2 = rd.sample(range(0, num_class_Y), foldsize_Y)
    #print(ry2)
    ry3 = rd.sample(range(0, num_class_Y), foldsize_Y)
    ry4 = rd.sample(range(0, num_class_Y), foldsize_Y)
    ry5 = rd.sample(range(0, num_class_Y), foldsize_Y)
    #print(ry1)
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
    #print(Y_1)
    #Y_1 = class_Y.iloc[0:(foldsize_Y), :]
    #Y_2 = class_Y.iloc[foldsize_Y: (foldsize_Y * 2), :]
    #Y_3 = class_Y.iloc[foldsize_Y * 2:(foldsize_Y * 3 ), :]
    #Y_4 = class_Y.iloc[foldsize_Y * 3:(foldsize_Y * 4 ), :]
    #Y_5 = class_Y.iloc[foldsize_Y * 4: (num_class_Y ), :]

#Split class N into 5 sets
    rn1 = rd.sample(range(0, num_class_N), foldsize_N)
    rn2 = rd.sample(range(0, num_class_N), foldsize_N)
    rn3 = rd.sample(range(0, num_class_N), foldsize_N)
    rn4 = rd.sample(range(0, num_class_N), foldsize_N)
    rn5 = rd.sample(range(0, num_class_N), foldsize_N)
    #print(rn1)
    N_1 = class_N.iloc[rn1, :]
    #print(N_1)

    #print(rn2)
    N_2 = class_N.iloc[rn2, :]
    N_3 = class_N.iloc[rn3, :]
    N_4 = class_N.iloc[rn4, :]
    N_5 = class_N.iloc[rn5, :]
    #print(N_2)
    #N_1 = class_N.iloc[0:(foldsize_N), :]
    #N_2 = class_N.iloc[foldsize_N: (foldsize_N * 2 ), :]
    #N_3 = class_N.iloc[foldsize_N * 2:(foldsize_N * 3 ), :]
    #N_4 = class_N.iloc[foldsize_N * 3:(foldsize_N * 4 ), :]
    #N_5 = class_N.iloc[foldsize_N * 4: (num_class_N ), :]
    #fold_size = int(num_instances / n_splits)

#Fold 1
    train_set1 = pd.concat([Y_2,Y_3,Y_4,Y_5, N_2,N_3,N_4,N_5], ignore_index=True)
    #print(train_set1.shape)
    #print(train_set1.iloc[1,:])
    #print(train_set1.shape)
    test_set1 = pd.concat([Y_1,N_1], ignore_index=True)
    #np.save('train_set1', train_set1)
    #np.save('test_set1', test_set1)
    #print(test_set1.shape)
    #print(num_class_Y)
    #print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    #print(num_class_N)
    #print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    #dataset_Y_pro,dataset_N_pro, pro_Ori, matrix, acc, pre, recall
    """
    print("Result for fold 1")
    dataset_Y_pro_1, dataset_N_pro_1,Pro_fold_1,acc1, precision1, recall1 = nb.NBRresult(train_set1,test_set1)
    dataset_Y_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_1.csv')
    dataset_N_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_1.csv')
    Pro_fold_1.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_1.csv')
    """


#Fold 2
    train_set2 = pd.concat([Y_1,Y_3,Y_4,Y_5, N_1,N_3,N_4,N_5], ignore_index=True)
    #print(train_set1.shape)
    test_set2 = pd.concat([Y_2,N_2], ignore_index=True)
    #np.save('train_set2', train_set2)
    #np.save('test_set2', test_set2)
    #print(test_set1.shape)
    #print(num_class_Y)
    #print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    #print(num_class_N)
    #print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    """
    print("Result for fold 2")
    dataset_Y_pro_2, dataset_N_pro_2, Pro_fold_2,acc2, precision2, recall2 = nb.NBRresult(train_set2,test_set2)
    dataset_Y_pro_2.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_2.csv')
    dataset_N_pro_2.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_2.csv')
    Pro_fold_2.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_2.csv')
"""
    #Fold 3
    train_set3 = pd.concat([Y_1,Y_2,Y_4,Y_5, N_1,N_2,N_4,N_5], ignore_index=True)
    #print(train_set1.shape)
    test_set3 = pd.concat([Y_3,N_3], ignore_index=True)
    #np.save('train_set3', train_set3)
    #np.save('test_set3', test_set3)
    #print(test_set1.shape)
    #print(num_class_Y)
    #print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    #print(num_class_N)
    #print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    """
    print("Result for fold 3")
    dataset_Y_pro_3, dataset_N_pro_3, Pro_fold_3,acc3, precision3, recall3 = nb.NBRresult(train_set3,test_set3)
    dataset_Y_pro_3.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_3.csv')
    dataset_N_pro_3.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_3.csv')
    Pro_fold_3.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_3.csv')
"""
#Fold 4
    train_set4 = pd.concat([Y_1,Y_2,Y_3,Y_5, N_1,N_2,N_3,N_5], ignore_index=True)
    #print(train_set1.shape)
    test_set4 = pd.concat([Y_4,N_4], ignore_index=True)
    #np.save('train_set4', train_set4)
    #np.save('test_set4', test_set4)
    #print(test_set1.shape)
    #print(num_class_Y)
    #print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    #print(num_class_N)
    #print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    """
    print("Result for fold 4")
    dataset_Y_pro_4, dataset_N_pro_4, Pro_fold_4,acc4, precision4, recall4 =nb.NBRresult(train_set4,test_set4)
    dataset_Y_pro_4.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_4.csv')
    dataset_N_pro_4.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_4.csv')
    Pro_fold_4.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_4.csv')
    """

#Fold 5
    train_set5 = pd.concat([Y_1,Y_2,Y_3,Y_4, N_1,N_2,N_3,N_4], ignore_index=True)
    #print(train_set1.shape)
    test_set5 = pd.concat([Y_5,N_5], ignore_index=True)
    #np.save('train_set5', train_set5)
    #np.save('test_set5', test_set5)
    #print(test_set1.shape)
    #print(num_class_Y)
    #print(Y_1.shape[0],Y_2.shape[0],Y_3.shape[0],Y_4.shape[0],Y_5.shape[0])
    #print(num_class_N)
    #print(N_1.shape[0], N_2.shape[0], N_3.shape[0], N_4.shape[0], N_5.shape[0])
    """
    print("Result for fold 5")
    dataset_Y_pro_5, dataset_N_pro_5, Pro_fold_5,acc5, precision5, recall5 = nb.NBRresult(train_set5,test_set5)
    dataset_Y_pro_5.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_5.csv')
    dataset_N_pro_5.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_5.csv')
    Pro_fold_5.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_5 .csv')
    """

#Average of acc, precision, recall
    """
    acc_avg = float((acc1+acc2+acc3+acc4+acc5)/5)
    precision_avg = float((precision1+precision2+precision3+precision4+precision5)/5)
    recall_avg = float((recall1+recall2+recall3+recall4+recall5)/5)
    print("Accuracy Ave: {}".format(acc_avg))
    print("Precision Ave: {}".format(precision_avg))
    print("Recall Ave: {}".format(recall_avg))
    """

    #return acc_avg,precision_avg,recall_avg


main()