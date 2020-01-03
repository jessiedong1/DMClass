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
    cww = cww[1:985]


    filename = r'D:\Uca\Thesis\NLP\OutputnoLabel.csv'

    dataset = load_file(filename)
    dataset = dataset.drop(columns=['Wine', 'Year', 'Score', 'Price', 'Country', 'Region', 'Issue Date'])
    #print(dataset.shape)
    #print(dataset)
    mapping(cww, dataset)

def load_file(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    return dataset

def mapping(cww, dataset):
    ROWS = dataset.shape[0]
    COLUMNS = len(cww['SUBCATEGORY_NAME'].unique())
    print(ROWS)
    print(COLUMNS)
    sub_names = cww.SUBCATEGORY_NAME.unique()
    print(sub_names)
    arr = [[0] * COLUMNS] * ROWS





main()