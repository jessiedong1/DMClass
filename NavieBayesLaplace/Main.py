import NavieBayes.NBClassifier as nb
import NavieBayes.Corss_Validation as cv

def main():
    filename = 'D:\Spring2019\DataMining\Dataset\Output.csv'
    # Load the data
    dataset = nb.loadCSV(filename)
    #print(dataset.shape)
    cv. Cross_Validation(dataset, 5)

main()