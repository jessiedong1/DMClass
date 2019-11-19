import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():

    filename = r'D:\Uca\Thesis\NLP\Nor.csv'
    dataset, X, y = loadCSV(filename)
    NaiveB(X,y)

#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    X = dataset.iloc[:,0:14]
    y = dataset.iloc[:,14]
    return dataset, X, y

def NaiveB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=52)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    pro = gnb.predict_log_proba(X_test)

    predicted_labels = gnb.predict(X_test)
    print(gnb.class_prior_)
    daf = pd.DataFrame(pro, columns=['Pro_N', 'Pro_Y'])
    daf['Pre'] = predicted_labels
    y_test1 = np.array(y_test)
    daf['Actual'] = y_test1
    print(daf)
    py = daf['Pro_Y']
    pn = daf['Pro_N']
    colors = ['navy', 'darkorange']
    target_names = [0, 1]
    labels = daf['Actual']
    print(y_test1)
    print(predicted_labels)
    tn, fp, fn, tp = confusion_matrix(y_test1, predicted_labels, labels=target_names).ravel()
    print('tp {}'.format(tp))
    print('fn {}'.format(fn))
    print('fp {}'.format(fp))
    print('tn {}'.format(tn))
    std = np.linspace(0, 1, 1000000)

    plt.scatter(py, pn, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(std, std + 0, linestyle='solid')
    # plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Probability_Y')
    plt.ylabel('Probability_N')

    plt.grid(True)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.legend()
    plt.show()

main()