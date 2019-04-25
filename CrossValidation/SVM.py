from NavieBayes.SVM import *
import numpy as np

def main():
    trainset1 = np.load('train_set1.npy')
    trainset2 = np.load('train_set2.npy')
    trainset3 = np.load('train_set3.npy')
    trainset4 = np.load('train_set4.npy')
    trainset5 = np.load('train_set5.npy')

    testset1 = np.load('test_set1.npy')
    testset2 = np.load('test_set2.npy')
    testset3 = np.load('test_set3.npy')
    testset4 = np.load('test_set4.npy')
    testset5 = np.load('test_set5.npy')


    # Fold 1
    trainset1_X, trainset1_y = split_x_y(trainset1)
    testset1_X, testset1_y = split_x_y(testset1)
    #print(trainset1_X.shape)
    #print(trainset1_y.shape)

    print("Fold 1:")
    acc1, precision1, recall1 = LinearSVM(trainset1_X, trainset1_y, testset1_X, testset1_y)

    # Fold 2
    trainset2_X, trainset2_y = split_x_y(trainset2)
    testset2_X, testset2_y = split_x_y(testset2)
    print("Fold 2:")
    acc2, precision2, recall2 = LinearSVM(trainset2_X, trainset2_y, testset2_X, testset2_y)

    # Fold 3
    trainset3_X, trainset3_y = split_x_y(trainset3)
    testset3_X, testset3_y = split_x_y(testset3)
    print("Fold 3:")
    acc3, precision3, recall3 = LinearSVM(trainset3_X, trainset3_y, testset3_X, testset3_y)

    # Fold 4
    trainset4_X, trainset4_y = split_x_y(trainset4)
    testset4_X, testset4_y = split_x_y(testset4)
    print("Fold 4:")
    acc4, precision4, recall4 = LinearSVM(trainset4_X, trainset4_y, testset4_X, testset4_y)

    # Fold 5
    trainset5_X, trainset5_y = split_x_y(trainset5)
    testset5_X, testset5_y = split_x_y(testset5)
    print("Fold 5:")
    acc5, precision5, recall5 = LinearSVM(trainset5_X, trainset5_y, testset5_X, testset5_y)

    acc_avg = float((acc1 + acc2 + acc3 + acc4 + acc5) / 5)
    precision_avg = float((precision1 + precision2 + precision3 + precision4 + precision5) / 5)
    recall_avg = float((recall1 + recall2 + recall3 + recall4 + recall5) / 5)
    print("Accuracy Ave: {:.4f}".format(acc_avg))
    print("Precision Ave: {:.4f}".format(precision_avg))
    print("Recall Ave: {:.4f}".format(recall_avg))

main()