import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import NavieBayes.NBClassifier as nb
from sklearn.decomposition import PCA
colors = ['navy', 'darkorange']
target_names = ['90-','90+']



def main():
    #X_PCA, y_PCA = load_dataset()
    #show_PCA(X_PCA,y_PCA)

    #show_NB()
    #plot_NB()
    Show_NBResults()


def load_dataset():
    dataset = dataset = pd.read_csv('D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv')
    dataset.head()
    dataset = pd.DataFrame(dataset)
    dataset['Label'] = np.where(dataset['Score'] < 90, 0, 1)
    X = dataset.iloc[:, 7:991]
    # print(X)
    # column_name = dataset.columns
    # print(column_name)
    y = dataset['Label']

    return X,y

def show_PCA(X,y):

    #print(y)
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    plt.figure()
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Wine Dataset')
    plt.show()

    return X,y
def show_NB():
    dataset = dataset = pd.read_csv('D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv')
    dataset.head()
    dataset = pd.DataFrame(dataset)
    dataset['Label'] = np.where(dataset['Score'] < 90, 0, 1)
    dataset_Y_pro_1, dataset_N_pro_1, Pro_fold_1, acc1, precision1, recall1 = nb.NBRresult(dataset, dataset)
    dataset_Y_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_Y_pro_1.csv')
    dataset_N_pro_1.to_csv(r'D:\Spring2019\DataMining\Output\Dataset_N_pro_1.csv')
    Pro_fold_1.to_csv(r'D:\Spring2019\DataMining\Output\Pro_fold_1.csv')

def plot_NB():
    dataset = pd.read_csv('D:\Spring2019\DataMining\Outputs\Dataset\Pro_fold_1.csv')

    X = dataset['Probability_Y']

    y = dataset['Probability_N']

    labels = dataset['Actual_Labels']

    std = np.linspace(0, 1, 1000000)
    # plt.titple("MLP Classifier")

    # plt.plot(X, y, 'ro', label='Train Precision')

    plt.scatter(X, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(std, std + 0, linestyle='solid')
    # plt.plot(epochs, val_pre, 'b', label="Test Precision")

    plt.xlabel('Probability_Y')
    plt.ylabel('Probability_N')
    plt.grid(True)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.legend()
    plt.show()


def Show_NBResults():
    dataset = pd.read_csv('D:\Spring2019\DataMining\Outputs\Random\Rd_laplace_90\Pro_fold_1.csv')

    #dataset.head()

    #print(dataset.columns)

    X = dataset['R_Y']

    y = dataset['Rf_N']
    #y = y.drop([194])

    #print(y)
   #labels = dataset['Predict_Class']
    labels = dataset['Actual_Labels']

    #labels = labels.drop([194])

    std = np.linspace(0, 1, 1000000)
    #plt.titple("MLP Classifier")

    #plt.plot(X, y, 'ro', label='Train Precision')

    plt.scatter(X,y,c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(std, std + 0, linestyle='solid')
    #plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Probability_Y')
    plt.ylabel('Probability_N')

    plt.grid(True)
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.legend()
    plt.show()



main()