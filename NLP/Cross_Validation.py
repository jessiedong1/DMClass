import pandas as pd
import numpy as np

import random as rd
import NLP.GaussianNB as gnb
import NLP.GaussianNBSub as gnbsub
import NLP.NBLaplace as nbl

results = pd.DataFrame()
results2 = pd.DataFrame()
results3= pd.DataFrame()
results4 = pd.DataFrame()
results5 = pd.DataFrame()

rd.seed(1)

def main():
    filename = r'D:\Uca\Thesis\NLP\Dataset\Wine1855_Category.csv'
    dataset_Category= loadCSV(filename)
    Cross_Validation_Category(dataset_Category, 5)

    filename = r'D:\Uca\Thesis\NLP\Dataset\Wine1855_SUB.csv'
    dataset_SUB = loadCSV(filename)
    Cross_Validation_Sub(dataset_SUB, 5)

    filename = r'D:\Uca\Thesis\NLP\Dataset\Wine1855.csv'
    dataset_Nor = loadCSV(filename)
    Cross_Validation_Nor(dataset_Nor, 5)

    results.to_csv('results.csv')
    results2.to_csv('results2.csv')
    results3.to_csv('results3.csv')
    results4.to_csv('results4.csv')
    results5.to_csv('results5.csv')


    # acc1, precision1, recall1 = Confusion_Matrix(results)
    # acc2, precision2, recall2 = Confusion_Matrix(results2)
    # acc3, precision3, recall3 = Confusion_Matrix(results3)
    # acc4, precision4, recall4 = Confusion_Matrix(results4)
    # acc5, precision5, recall5 = Confusion_Matrix(results5)
    #
    #
    # print("Accuracy: {:.4f}".format((acc1+acc2+acc3+acc4+acc5)/5))
    # print("Precision: {:.4f}".format((precision1 + precision2 + precision3 + precision4 + precision5) / 5))
    # print("Recall: {:.4f}".format((recall1 + recall2 + recall3 + recall4 + recall5) / 5))


#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    dataset['Label'] = np.where(dataset['Score'] < 90, 0, 1)
    #Min-max normlization
    COLUMNS = dataset.shape[1]
    dataset_attributes = dataset.iloc[:,4:COLUMNS]
    normalized_df = (dataset_attributes - dataset_attributes.min()) / (dataset_attributes.max() - dataset_attributes.min())
    normalized_df['Label'] = dataset['Label']
    return normalized_df

#Seperate Class by 'Label'
def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))
    #class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

#Making prediction - majority voting + confision matrix
def Confusion_Matrix(pro_Ori):
    #Majority voting
    # pro_Ori['Total'] = pro_Ori.iloc[:, 0:3].sum(axis=1)
    # predict = [99]*pro_Ori.shape[0]
    # for i in range(pro_Ori.shape[0]):
    #     if pro_Ori['Total'][i] >= 2:
    #         predict[i] = 1
    #     else:
    #         predict[i] = 0
    #
    # pro_Ori['Predict_Class'] = predict
    #print(pro_Ori)
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(pro_Ori.shape[0]):
        if ((pro_Ori.Predicted_Sub[i] == 1) and (pro_Ori.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((pro_Ori.Predicted_Sub[i] == 0) and (pro_Ori.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((pro_Ori.Predicted_Sub[i] == 1) and (pro_Ori.Actual_Labels[i] == 0)):
            fp = fp + 1
        else:
            tn = tn + 1

    array = {'Predicted Class = 90+': [tp, fp],
             'Predicted Class = 90-': [fn, tn]}

    actual_Class = ['Actual Class = 90+ ', 'Actual Class = 90-']
    # Put two list into a dataframe
    matrix = pd.DataFrame(array, actual_Class)
    # matrix.to_csv(r'D:\Spring2019\DataMining\Output\Con_Matrix.csv')
    print("Confusion Matrix: ")
    print(matrix)
    precision = 0
    recall = 0
    acc = float((tp + tn) / (tp + fn + fp + tn))
    print("Accuracy: {:.4f}".format(acc))
    if (tp != 0):
        precision = float(tp / (tp + fp))
        recall = float(tp / (tp + fn))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
    print()

    return acc, precision, recall

def Cross_Validation_Nor(dataset, n_splits):
    # num_instances = dataset.shape[0]
    # train_set = pd.DataFrame()
    # dataset_copy = dataset.copy()
    class_Y, class_N = Group_Class(dataset)
    num_class_Y = class_Y.shape[0]
    num_class_N = class_N.shape[0]
    #print(num_class_Y, num_class_N)
    # print("90+ wines are %d, 90- %d" %(num_class_Y, num_class_N))
    # print(4263/(4263+10086))
    # print(10086/(4263+10086))

    foldsize_Y = int(num_class_Y / n_splits)
    #print(foldsize_Y)
    foldsize_N = int(num_class_N / n_splits)
    #print(foldsize_N)

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

    # Fold 1

    Nor_Result = nbl.NBRresult(train_set1, test_set1)
    results['Pro_Y_Nor'] = Nor_Result['Probability_Y']
    results['Pro_N_Nor'] = Nor_Result['Probability_N']
    results['Predicted_Nor'] = Nor_Result['Predict_Class']
    results['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = nbl.NBRresult(train_set2, test_set2)
    results2['Pro_Y_Nor'] = Nor_Result['Probability_Y']
    results2['Pro_N_Nor'] = Nor_Result['Probability_N']
    results2['Predicted_Nor'] = Nor_Result['Predict_Class']
    results2['Actual_Labels'] = Nor_Result['Actual_Labels']


    Nor_Result = nbl.NBRresult(train_set3, test_set3)
    results3['Pro_Y_Nor'] = Nor_Result['Probability_Y']
    results3['Pro_N_Nor'] = Nor_Result['Probability_N']
    results3['Predicted_Nor'] = Nor_Result['Predict_Class']
    results3['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = nbl.NBRresult(train_set4, test_set4)
    results4['Pro_Y_Nor'] = Nor_Result['Probability_Y']
    results4['Pro_N_Nor'] = Nor_Result['Probability_N']
    results4['Predicted_Nor'] = Nor_Result['Predict_Class']
    results4['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = nbl.NBRresult(train_set5, test_set5)
    results5['Pro_Y_Nor'] = Nor_Result['Probability_Y']
    results5['Pro_N_Nor'] = Nor_Result['Probability_N']
    results5['Predicted_Nor'] = Nor_Result['Predict_Class']
    results5['Actual_Labels'] = Nor_Result['Actual_Labels']

    print(results)

def Cross_Validation_Sub(dataset, n_splits):
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
    #print(foldsize_Y)
    foldsize_N = int(num_class_N / n_splits)
    #print(foldsize_N)

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

    # Fold 1

    print("Fold 1: ")
    Sub_Result = gnbsub.NormalDisMin(train_set1, test_set1)
    results['Pro_Y_Sub'] = Sub_Result['Probability_Y']
    results['Pro_N_Sub'] = Sub_Result['Probability_N']
    results['Predicted_Sub'] = Sub_Result['Predict_Class']
    #results['Actual_Labels'] = Sub_Result['Actual_Labels']

    Nor_Result = gnbsub.NormalDisMin(train_set2, test_set2)
    results2['Pro_Y_Sub'] = Nor_Result['Probability_Y']
    results2['Pro_N_Sub'] = Nor_Result['Probability_N']
    results2['Predicted_Sub'] = Nor_Result['Predict_Class']
   # results2['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = gnbsub.NormalDisMin(train_set3, test_set3)
    results3['Pro_Y_Sub'] = Nor_Result['Probability_Y']
    results3['Pro_N_Sub'] = Nor_Result['Probability_N']
    results3['Predicted_Sub'] = Nor_Result['Predict_Class']
    #results3['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = gnbsub.NormalDisMin(train_set4, test_set4)
    results4['Pro_Y_Sub'] = Nor_Result['Probability_Y']
    results4['Pro_N_Sub'] = Nor_Result['Probability_N']
    results4['Predicted_Sub'] = Nor_Result['Predict_Class']
    #results4['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = gnbsub.NormalDisMin(train_set5, test_set5)
    results5['Pro_Y_Sub'] = Nor_Result['Probability_Y']
    results5['Pro_N_Sub'] = Nor_Result['Probability_N']
    results5['Predicted_Sub'] = Nor_Result['Predict_Class']
    #results5['Actual_Labels'] = Nor_Result['Actual_Labels']

def Cross_Validation_Category(dataset, n_splits):
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


    Category_Result = gnb.NormalDisMin(train_set1, test_set1)
    results['Pro_Y_Category'] = Category_Result['Probability_Y']
    results['Pro_N_Category'] = Category_Result['Probability_N']
    results['Predicted_Category'] = Category_Result['Predict_Class']
    #results['Actual_Labels'] = Category_Result['Actual_Labels']

    Nor_Result = gnb.NormalDisMin(train_set2, test_set2)
    results2['Pro_Y_Category'] = Nor_Result['Probability_Y']
    results2['Pro_N_Category'] = Nor_Result['Probability_N']
    results2['Predicted_Category'] = Nor_Result['Predict_Class']
    #results2['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = gnb.NormalDisMin(train_set3, test_set3)
    results3['Pro_Y_Category'] = Nor_Result['Probability_Y']
    results3['Pro_N_Category'] = Nor_Result['Probability_N']
    results3['Predicted_Category'] = Nor_Result['Predict_Class']
    #results3['Actual_Labels'] = Nor_Result['Actual_Labels']

    Nor_Result = gnb.NormalDisMin(train_set4, test_set4)
    results4['Pro_Y_Category'] = Nor_Result['Probability_Y']
    results4['Pro_N_Category'] = Nor_Result['Probability_N']
    results4['Predicted_Category'] = Nor_Result['Predict_Class']
    #results4['Actual_Labels'] = Nor_Result['Actual_Labels']


    Nor_Result = gnb.NormalDisMin(train_set5, test_set5)
    results5['Pro_Y_Category'] = Nor_Result['Probability_Y']
    results5['Pro_N_Category'] = Nor_Result['Probability_N']
    results5['Predicted_Category'] = Nor_Result['Predict_Class']
    #results5['Actual_Labels'] = Nor_Result['Actual_Labels']





    #
    # # Fold 1
    # print("Fold2: ")
    # acc2, precision2, recall2 = gnb.NormalDisMean(train_set2, test_set2)
    # acc2min, precision2min, recall2min = gnb.NormalDisMin(train_set2, test_set2)
    #
    # # Fold 1
    # print("Fold3: ")
    # acc3, precision3, recall3 = gnb.NormalDisMean(train_set3, test_set3)
    # acc3min, precision3min, recall3min = gnb.NormalDisMin(train_set3, test_set3)
    # # Fold 1
    # print("Fold4: ")
    # acc4, precision4, recall4 = gnb.NormalDisMean(train_set4, test_set4)
    # acc4min, precision4min, recall4min = gnb.NormalDisMin(train_set4, test_set4)
    #
    # # Fold 1
    # print("Fold5 : ")
    # acc5, precision5, recall5 = gnb.NormalDisMean(train_set5, test_set5)
    # acc5min, precision5min, recall5min = gnb.NormalDisMin(train_set5, test_set5)


    # print("Accuracy mean: {:.4f}".format((acc1+acc2+acc3+acc4+acc5)/5))
    # print("Precision mean: {:.4f}".format((precision1 + precision2 + precision3 + precision4 + precision5) / 5))
    # print("Recall mean: {:.4f}".format((recall1 + recall2 + recall3 + recall4 + recall5) / 5))
    #
    # print("Accuracy min: {:.4f}".format((acc1min + acc2min + acc3min + acc4min + acc5min) / 5))
    # print("Precision min: {:.4f}".format((precision1min + precision2min + precision3min + precision4min + precision5min) / 5))
    # print("Recall min: {:.4f}".format((recall1min + recall2min + recall3min + recall4min + recall5min) / 5))




main()