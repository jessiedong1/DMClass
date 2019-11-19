import NavieBayes.NBClassifier as nb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from NavieBayes.MLP import *
def main():
    print()
    filename = r'D:\UCA\THESIS\BordeauxWines.csv'
    # Load the data
    dataset = nb.loadCSV(filename)
    #print(dataset.columns.values)
    dataset_X, dataset_y = split_x_y(dataset)
    #print(dataset_X.columns.values)
    #print(dataset_y.columns.values)

    # test_size: what proportion of original data is used for test set
    train_x, test_x, train_y, test_y = train_test_split(dataset_X, dataset_y, test_size=0.3,random_state=0)
    #print(train_x.shape)
    #print(train_y.shape)

    #print(pca_0_1_df)
    train_x_pca, train_y_pca = PCA_re(train_x,train_y)
    test_x_pca, test_y_pca = PCA_re(test_x, test_y)
    val_acc1, val_pre1, val_recall1 = Cro_val(train_x_pca, train_y_pca, test_x_pca, test_y_pca)

def split_x_y(dataset):
    columns = dataset.shape[1]-1
    #print(dataset)
    x = dataset.iloc[:, 4: columns]
    x = pd.DataFrame(x)
    #print(x[:,983])
    y = dataset.iloc[:,columns]
    y = pd.DataFrame(y)
    #print(y)
    return x,y

def PCA_re(train_x, train_y):
    pca = PCA(n_components=50)
    pca.fit(train_x)
    # print(pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    # plt.show()
    pca_0_1_df = pd.DataFrame(pca.transform(train_x), columns=['PCA%i' % i for i in range(50)], index=train_x.index)
    pca_0_1_df['Label'] = train_y
    train_x_pca = pca_0_1_df.iloc[:, 0:50]

    train_y_pca = pca_0_1_df['Label']

    return train_x_pca, train_y_pca

#main()