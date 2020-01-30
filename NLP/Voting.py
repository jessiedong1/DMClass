"""
Jessie - 01/2020
This class demonstrates the decision making based on different voters
Voter1 = Category
Voter2 = Subcatogory
Voter3 = Normalized
Voter4 = V1*V2*V3
Voter5 = V1*V2
Voter6 = V2*V3
Voter7 = V1*V3
"""
import pandas as pd
import numpy as np
import random as rd

# result1 = pd.DataFrame()
# result2 = pd.DataFrame()
# result3= pd.DataFrame()
# result4 = pd.DataFrame()
# result5 = pd.DataFrame()

def main():
    filename = 'results.csv'
    result1 = Load_csv(filename)
    filename = 'results2.csv'
    result2 = Load_csv(filename)
    filename = "results3.csv"
    result3 = Load_csv(filename)
    filename = "results4.csv"
    result4 = Load_csv(filename)
    filename = "results5.csv"
    result5 = Load_csv(filename)

#Voter 4
    acc1, precision1, recall1 = voter4(result1)
    acc2, precision2, recall2 = voter4(result2)
    acc3, precision3, recall3 = voter4(result3)
    acc4, precision4, recall4 = voter4(result4)
    acc5, precision5, recall5 = voter4(result5)

#Voter 5
    acc1, precision1, recall1 = voter5(result1)
    acc2, precision2, recall2 = voter5(result2)
    acc3, precision3, recall3 = voter5(result3)
    acc4, precision4, recall4 = voter5(result4)
    acc5, precision5, recall5 = voter5(result5)

    # Voter 6
    acc1, precision1, recall1 = voter6(result1)
    acc2, precision2, recall2 = voter6(result2)
    acc3, precision3, recall3 = voter6(result3)
    acc4, precision4, recall4 = voter6(result4)
    acc5, precision5, recall5 = voter6(result5)


    # Voter 7
    acc1, precision1, recall1 = voter7(result1)
    acc2, precision2, recall2 = voter7(result2)
    acc3, precision3, recall3 = voter7(result3)
    acc4, precision4, recall4 = voter7(result4)
    acc5, precision5, recall5 = voter7(result5)

    acc1, precision1, recall1 = Making_Decision(result1)
    acc2, precision2, recall2 = Making_Decision(result2)
    acc3, precision3, recall3 = Making_Decision(result3)
    acc4, precision4, recall4 = Making_Decision(result4)
    acc5, precision5, recall5 = Making_Decision(result5)


    print("Accuracy: {:.4f}".format((acc1+acc2+acc3+acc4+acc5)/5))
    print("Precision: {:.4f}".format((precision1 + precision2 + precision3 + precision4 + precision5) / 5))
    print("Recall: {:.4f}".format((recall1 + recall2 + recall3 + recall4 + recall5) / 5))

def Load_csv(filename):
    result = pd.read_csv(filename)
    result.head()
    result = pd.DataFrame(result)
    return result

def Making_Decision(result1):

    result1['Total'] = result1['Predicted_Category'] + result1['Predicted_Sub'] + result1['Predicted_Nor'] + result1['v4'] + result1['v5'] +result1['v6'] +result1['v7']
    result1['Predict_Total'] = np.where(result1['Total'] >= 4, 1, 0)

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(result1.shape[0]):
        if ((result1.Predict_Total[i] == 1) and (result1.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((result1.Predict_Total[i] == 0) and (result1.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((result1.Predict_Total[i] == 1) and (result1.Actual_Labels[i] == 0)):
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

def voter7(result1):
    result1['v7_Y'] = result1['Pro_Y_Category']*  result1['Pro_Y_Nor']
    result1['v7_N'] = result1['Pro_N_Category'] *  result1['Pro_N_Nor']
    result1['v7'] = np.where(result1['v7_Y'] > result1['v7_N'],1,0 )
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(result1.shape[0]):
        if ((result1.v7[i] == 1) and (result1.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((result1.v7[i] == 0) and (result1.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((result1.v7[i] == 1) and (result1.Actual_Labels[i] == 0)):
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

def voter6(result1):
    result1['v6_Y'] = result1['Pro_Y_Sub'] * result1['Pro_Y_Nor']
    result1['v6_N'] = result1['Pro_N_Sub'] * result1['Pro_N_Nor']
    result1['v6'] = np.where(result1['v6_Y'] > result1['v6_N'],1,0 )
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(result1.shape[0]):
        if ((result1.v6[i] == 1) and (result1.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((result1.v6[i] == 0) and (result1.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((result1.v6[i] == 1) and (result1.Actual_Labels[i] == 0)):
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

def voter5(result1):
    result1['v5_Y'] = result1['Pro_Y_Category']* result1['Pro_Y_Sub']
    result1['v5_N'] = result1['Pro_N_Category'] * result1['Pro_N_Sub']
    result1['v5'] = np.where(result1['v5_Y'] > result1['v5_N'],1,0 )
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(result1.shape[0]):
        if ((result1.v5[i] == 1) and (result1.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((result1.v5[i] == 0) and (result1.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((result1.v5[i] == 1) and (result1.Actual_Labels[i] == 0)):
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

def voter4(result1):
    result1['v4_Y'] = result1['Pro_Y_Category']* result1['Pro_Y_Sub'] * result1['Pro_Y_Nor']
    result1['v4_N'] = result1['Pro_N_Category'] * result1['Pro_N_Sub'] * result1['Pro_N_Nor']
    result1['v4'] = np.where(result1['v4_Y'] > result1['v4_N'],1,0 )
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(result1.shape[0]):
        if ((result1.v4[i] == 1) and (result1.Actual_Labels[i] == 1)):
            tp = tp + 1
        elif ((result1.v4[i] == 0) and (result1.Actual_Labels[i] == 1)):
            fn = fn + 1
        elif ((result1.v4[i] == 1) and (result1.Actual_Labels[i] == 0)):
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



main()