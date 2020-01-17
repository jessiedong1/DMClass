import pandas as pd
import numpy as np
import random as rd

def main():
    cww = r'D:\Uca\Thesis\NLP\CWW.csv'
    cww = load_file(cww)

    cww = cww.drop(['SPECIFIC_NAME'],axis =1)
    #print(cww['CATEGORY_NAME'].unique())
    #print(cww.groupby('CATEGORY_NAME').count())
    #cww = cww[1:985]
    #printCWW
    cww = cww.drop_duplicates('NORMALIZED_NAME')

    #cww = cww[1:985]

    #filename = r'D:\Uca\Thesis\NLP\Dataset\Wine1855.csv'
    filename = r'D:\Uca\Thesis\NLP\Dataset\BordeauxWines.csv'
    dataset = load_file(filename)

    # print(dataset.columns)
    # print(cww.columns)
    #print(dataset.shape)
    #print(dataset)
    mapping(cww, dataset)

def load_file(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    return dataset

def mapping(cww, dataset):
    dataset = dataset.drop(columns=['Name', 'Year', 'Score', 'Price'])
    ROWS = dataset.shape[0]
    COLUMNS = len(cww['CATEGORY_NAME'].unique())
    print(ROWS)
    print(COLUMNS)

    #sub_names = cww.SUBCATEGORY_NAME.unique()
    sub_names = cww['CATEGORY_NAME'].unique()
    arr = [[0] * COLUMNS] * ROWS
    arr = pd.DataFrame(arr)
    arr.columns = sub_names
    #arr.at[0,sub_names[0]] = 5
    counts = cww.groupby('CATEGORY_NAME').count()
    sub_counts = [0]*COLUMNS

    for i in range(COLUMNS):
        for j in range(COLUMNS):
            if sub_names[i] == counts.index.values[j]:
                sub_counts[i] = counts.iloc[j,0]

    sub_counts = pd.DataFrame(sub_counts, index = sub_names)
    # print(sub_names)
    #print(sub_counts.iloc[10,0])
    print(sub_counts)
    a = 0
    for l in range(COLUMNS):
        print(a)
        b = sub_counts.iloc[l,0]+a
        print(b)
        arr.iloc[:,l] = dataset.iloc[:,a:b].sum(axis=1)
        #print(dataset.iloc[:,a:b].sum(axis=1))
        a = b


    #print(dataset.columns)

    #arr.to_csv(r'D:\Uca\Thesis\NLP\Dataset\BordeauxWines_Catrgory.csv')







main()