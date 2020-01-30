import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

def main():

    filename = r'D:\Uca\Thesis\NLP\Dataset\Wine1855_SUB.csv'
    dataset = loadCSV(filename)
    dataset_train = dataset[200:1000]
    dataset_test = dataset[:7]
    #print(dataset_test)
    pro_oris = NormalDisMin(dataset_train,dataset_test)
    print(pro_oris)

    #Confusion_Matrix(pro_oris)


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
    normalized_df = normalized_df.fillna(0)
    return normalized_df


    #return dataset

def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))

    # class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

def Zero_Frequency_mean(dataset):
    for i in range(dataset.shape[0]-1):
        if dataset.iloc[i] == 0:
            dataset.iloc[i] = dataset.mean()

    return dataset

def Zero_Frequency_min(dataset):
    min = dataset[dataset > .01].min()
    for i in range(dataset.shape[0]):
        if dataset.iloc[i] == 0:
            dataset.iloc[i] = min

    return dataset

def NormalDisMin(train_set, test_set):
    train_set_Yes, train_set_No = Group_Class(train_set)
    train_set_Yes_X, train_set_Yes_Y = Split_X_y(train_set)
    train_set_No_X, train_set_No_Y = Split_X_y(test_set)

    train_set_Yes_X_mean = train_set_Yes_X.mean()
    train_set_Yes_X_var = train_set_Yes_X.var()
    train_set_No_X_mean = train_set_No_X.mean()
    train_set_No_X_var = train_set_No_X.var()


    # print("Mean in trainset 90+ class")
    # print(train_set_Yes_X_mean)
    # print("Variance in train set 90+ Class")
    # print(train_set_Yes_X_var)
    # print("Mean in trainset 90- class")
    # print(train_set_No_X_mean)
    # print("Variance in train set 90- Class")
    # print(train_set_No_X_var)
    # print()

    #Handle zero frequency by assign the average
    train_set_Yes_mean = Zero_Frequency_min(train_set_Yes_X_mean)
    train_set_Yes_var = Zero_Frequency_min(train_set_Yes_X_var)
    train_set_No_mean = Zero_Frequency_min(train_set_No_X_mean)
    train_set_No_var = Zero_Frequency_min(train_set_No_X_var)

    # print("Mean in trainset 90+ class")
    # print(train_set_Yes_X_mean)
    # print("Variance in train set 90+ Class")
    # print(train_set_Yes_X_var)
    # print("Mean in trainset 90- class")
    # print(train_set_No_X_mean)
    # print("Variance in train set 90- Class")
    # print(train_set_No_X_var)

    ROWS = test_set.shape[0]
    COLS = test_set.shape[1]-1

    pro_yes = [[0] * COLS] * ROWS
    pro_yes = pd.DataFrame(pro_yes, columns=test_set.columns[0:COLS], dtype=float)
    pro_no = [[0] * COLS] * ROWS
    pro_no = pd.DataFrame(pro_no, columns=test_set.columns[0:COLS], dtype=float)
    for i in range(test_set.shape[0]):
        for j in range(COLS):
            pro_yes.iloc[i][j] = (1/(math.sqrt(2*math.pi*train_set_Yes_var[j])))*(math.exp(-((test_set.iloc[i][j]-train_set_Yes_mean[j])**2)/(2*train_set_Yes_var[j])))
            pro_no.iloc[i][j] = (1 / (math.sqrt(2 * math.pi * train_set_No_var[j]))) * (math.exp(-((test_set.iloc[i][j] - train_set_No_mean[j]) ** 2) / (2 * train_set_No_var[j])))

    pro_label_yes = pro_yes.product(axis =1)
    pro_label_no = pro_no.product(axis =1)

    Total_sample = train_set.shape[0]

    num_Y = train_set_Yes.shape[0]
    num_N = train_set_No.shape[0]
    # Calculate class distrubution
    pro_Y = float(num_Y / Total_sample)
    pro_N = float(num_N / Total_sample)


    pros_Y = pro_label_yes * pro_Y
    pro_Ori = pd.DataFrame(pros_Y.values, columns=['Probability_Y'])
    pros_N = pro_N *pro_label_no
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
    pro_Ori['Actual_Labels'] = test_set['Label'].values
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

def Split_X_y(dataset):
    cols = dataset.shape[1]-1
    X = dataset.iloc[:, 0:cols]
    y = dataset.iloc[:, cols]

    return X, y

#main()
