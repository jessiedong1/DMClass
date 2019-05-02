import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import NavieBayes.NBClassifier as nb
from sklearn.decomposition import PCA
colors = ['navy', 'darkorange']
target_names = ['90-','90+']



def main():

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

def Show_NBResults():
    dataset = pd.read_csv('D:\Spring2019\DataMining\Output\Pro_fold_1.csv')

    #dataset.head()

    #print(dataset.columns)

    X = dataset['Probability_Y']

    y = dataset['Probability_N']
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