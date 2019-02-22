import pandas as pd

def main():
    filename = 'D:\Spring2018\Artificial Intelligence\Lymph'
    loadCSV(filename)

def loadCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)

main()

