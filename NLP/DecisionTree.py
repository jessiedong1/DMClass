import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():

    filename = r'D:\Uca\Thesis\NLP\Nor.csv'
    dataset, X, y = loadCSV(filename)


#Load CSV File
def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    X = dataset.iloc[:,0:14]
    y = dataset.iloc[:,14]
    return dataset, X, y

# def GaussionNB(X_train,y_train,X_test,y_test):
#     # Seperate Class by 'Label'
#     X_train_yes, X_train_no = Group_Class(X_train)
#
#
#
#
# def Group_Class(dataset):
#     class_N, class_Y = (g for _, g in dataset.groupby('Label'))
#     # class_N, class_Y = dataset.groupby('Label')
#     return [class_Y, class_N]

def NaiveB(X_train,y_train,X_test,y_test):


    gnb = tree.DecisionTreeClassifier()
    gnb.fit(X_train, y_train)
    pro = gnb.predict_log_proba(X_test)
    #print(pro)

    predicted_labels = gnb.predict(X_test)
    #print(gnb.class_prior_)
    daf = pd.DataFrame(pro, columns=['Pro_N', 'Pro_Y'])
    daf['Pre'] = predicted_labels
    y_test1 = np.array(y_test)
    daf['Actual'] = y_test1
    #print(daf)
    py = daf['Pro_Y']
    pn = daf['Pro_N']
    colors = ['navy', 'darkorange']
    target_names = [0, 1]
    labels = daf['Actual']
    #print(y_test1)
    #print(predicted_labels)
    tn, fp, fn, tp = confusion_matrix(y_test1, predicted_labels, labels=target_names).ravel()
    print('tp {}'.format(tp), end = " ")
    print('fn {}'.format(fn), end = " ")
    print('fp {}'.format(fp), end = " ")
    print('tn {}'.format(tn), end = " ")
    print()
    acc, precision, recall = metricx(tp,fn,fp,tn)
    std = np.linspace(0, 1, 1000000)

    plt.scatter(py, pn, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(std, std + 0, linestyle='solid')
    # plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Probability_Y')
    plt.ylabel('Probability_N')

    plt.grid(True)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.legend()
    #plt.show()
    from sklearn.tree import export_graphviz
    export_graphviz(gnb, out_file='tree.dot', class_names=['90-', '90+'], feature_names=X_train.columns.values[0:13],
                    impurity=False, filled=True)

    import graphviz
    with open('tree.dot') as f:
        dot_graph = f.read()

    graph = graphviz.Source(dot_graph)
    graph.view()

    return acc, precision, recall

def metricx(tp, fn, fp,tn):
    acc = (tp+tn)/((tp+tn+fn+fp))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print('acc {}'.format(acc), end = " ")
    print('recll {}'.format(recall), end = " ")
    print('precision {}'.format(precision), end = " ")
    print()

    return acc, precision,recall

#main()