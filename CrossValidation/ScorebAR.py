import NavieBayes.NBClassifier as nb
import matplotlib.pyplot as plt

filename = r'D:\Uca\Thesis\BordeauxWines.csv'
# Load the data
dataset = nb.loadCSV(filename)
#score = dataset['Score']

# dataset['Year'].value_counts().plot(kind='bar')
# dataset.boxplot(column = 'Score')
#score_bar= score.value_counts()
#score_bar = pd.DataFrame(score_bar)


dataset.groupby(['Score']).count()['Label'].plot(kind='bar')

#plt.show()
print(dataset.columns)
print(dataset.shape)