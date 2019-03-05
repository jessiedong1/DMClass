"""
Navie Bayes with Laplace estimation
"""
import pandas as pd
import random as rd
import NavieBayes.NBClassifier as nb

# Calculate the probablity of one attributs in one class
def Cal_Pro_att_Lap(class_Y, att):
    # print(class_Y)
    # att_0, att_1= (g for _, g in class_Y.groupby(att))
    # print(att_0, att_1)
    data = class_Y[att]
    att_0 = 0
    att_1 = 0
    for i in range((class_Y.shape[0])):
        if data.iat[i] == 0:
            att_0 += 1
        else:
            att_1 += 1
    pro_0 = float((att_0 +1)/ (len(class_Y)+2))
    pro_1 = float((att_1+1) / (len(class_Y)+2))
    return pro_0, pro_1


def Cal_All_Attributs_Lap(class_Y, atts):
    size = len(atts)
    pro_0 = [0] * size
    pro_1 = [0] * size
    for i in range(size):
        pro_0[i], pro_1[i] = Cal_Pro_att_Lap(class_Y, atts[i])
        # pro_1[i] = Cal_Pro_att1(class_Y,atts[i])
    array = {'ATT=0': pro_0,
             'ATT=1': pro_1}
    ar = pd.DataFrame(array, atts)
    return ar

# Print accuracy, recall and precision

# Get the result
def NBRresult(train_set, test_set):
    rd.seed(1)
    # print(filename)

    # print(dataset.iat[0,0])
    # Extract the attributes to be calculated
    atts = train_set.columns.values
    # print(atts)
    atts = atts[8:((len(atts)) - 1)]

    Total_sample = train_set.shape[0]
    train_set_Y, train_set_N = nb.Group_Class(train_set)
    num_Y = train_set_Y.shape[0]
    num_N = train_set_N.shape[0]
    # Calculate class distrubution
    pro_Y = float(num_Y / Total_sample)
    pro_N = float(num_N / Total_sample)


    # Calculating all the attributes probabliaty in Y/N class without any fix
    dataset_Y_pro = Cal_All_Attributs_Lap(train_set_Y, atts)
    dataset_N_pro = Cal_All_Attributs_Lap(train_set_N, atts)
    # print(dataset_Y_pro)
    # print(dataset_N_pro)

    # Initilize test dataset
    # test_data = dataset.iloc[50:250, :]
    test_data_Labels = test_set['Label']
    # Get the NB in Y class
    test_pro_Y = nb.Get_Pro(test_set, dataset_Y_pro, atts)
    # Get the NB in N class
    test_pro_N = nb.Get_Pro(test_set, dataset_N_pro, atts)

    # print(test_pro)

    # Calculate NB
    pro_Lap = nb.Classify_NB(test_pro_Y, test_pro_N, pro_Y, pro_N)
    pro_Lap['Actual_Labels'] = test_data_Labels.values
    # Print the Final probablity
    # print(pro_Ori)
    nb.Confusion_Matrix(pro_Lap)



