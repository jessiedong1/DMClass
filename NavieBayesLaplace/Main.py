import numpy as np
import pandas as pd



def main():
    filename = 'D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv'
    # Load the data
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    dataset['Label'] = np.where(dataset['Score'] < 91, 0, 1)
    print(dataset)
    #print(dataset.groupby('Label').count())



main()