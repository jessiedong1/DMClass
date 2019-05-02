import NavieBayes.NBClassifier as nb
import matplotlib.pyplot as plt

filename = 'D:\Spring2019\DataMining\Dataset\OutputnoLabel.csv'
# Load the data
dataset = nb.loadCSV(filename)

#score = dataset['Score']

dataset['Score'].value_counts().plot(kind='bar')
#score_bar= score.value_counts()
#score_bar = pd.DataFrame(score_bar)
plt.show()




