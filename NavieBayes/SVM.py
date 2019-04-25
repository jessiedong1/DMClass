from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
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

#Fold 1
    trainset1_X, trainset1_y = split_x_y(trainset1)
    testset1_X, testset1_y = split_x_y(testset1)
    print("Fold 1:")
    acc1, precision1, recall1 =LinearSVM(trainset1_X,trainset1_y,testset1_X, testset1_y)

#Fold 2
    trainset2_X, trainset2_y = split_x_y(trainset2)
    testset2_X, testset2_y = split_x_y(testset2)
    print("Fold 2:")
    acc2, precision2, recall2 =LinearSVM(trainset2_X,trainset2_y,testset2_X, testset2_y)

#Fold 3
    trainset3_X, trainset3_y = split_x_y(trainset3)
    testset3_X, testset3_y = split_x_y(testset3)
    print("Fold 3:")
    acc3, precision3, recall3 = LinearSVM(trainset3_X,trainset3_y,testset3_X, testset3_y)

# Fold 4
    trainset4_X, trainset4_y = split_x_y(trainset4)
    testset4_X, testset4_y = split_x_y(testset4)
    print("Fold 4:")
    acc4, precision4, recall4 =LinearSVM(trainset4_X, trainset4_y, testset4_X, testset4_y)

# Fold 5
    trainset5_X, trainset5_y = split_x_y(trainset5)
    testset5_X, testset5_y = split_x_y(testset5)
    print("Fold 5:")
    acc5, precision5, recall5 =LinearSVM(trainset5_X, trainset5_y, testset5_X, testset5_y)

    acc_avg = float((acc1 + acc2 + acc3 + acc4 + acc5) / 5)
    precision_avg = float((precision1 + precision2 + precision3 + precision4 + precision5) / 5)
    recall_avg = float((recall1 + recall2 + recall3 + recall4 + recall5) / 5)
    print("Accuracy Ave: {:.4f}".format(acc_avg))
    print("Precision Ave: {:.4f}".format(precision_avg))
    print("Recall Ave: {:.4f}".format(recall_avg))
def split_x_y(dataset):
    columns = dataset.shape[1] - 1
    # print(dataset)
    x = dataset[:, 7: columns]
    #print(x.shape)
    # print(x[:,983])
    y = dataset[:, columns]
    #print(y)
    #print(y.dtype)
    y = y.astype('int')

    return x, y


def LinearSVM(X_train,y_train, X_test,y_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    tn, fp, fn, tp= confusion_matrix(y_test, y_pred).ravel()
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

    #y_predict = clf.predict(X_test)
    #print(y_predict)
    #clf_predictions = clf.predict(X_test)
    #print("Accuracy: {}%".format(clf.score(X_test, y_test)))
    #print("Accuracy: {}%".format(clf.score(X_test, y_test)))
    return acc, precision, recall




#main()




"""
    X_new = np.hstack([X, X[:, 1:] ** 2])
    figure = plt.figure()
    ax = Axes3D(figure,elev = -152, azim = -26)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_zscale('log')
    ax.set_xlabel("Pro_Y")
    ax.set_ylabel('Pro_N')
    ax.set_zlabel('**2')
    mask = y == 0
    ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',marker = '^', cmap=mglearn.cm2, s=60)
"""
