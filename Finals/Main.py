import numpy as np
from matplotlib import pyplot as plt
from Finals.BiMax import BiMax

generator = np.random.RandomState(1)
#data = generator.binomial(1, 0.5, (20, 20))
#print(data)

data = [[1,0,1,0],
        [1,1,0,1],
        [0,1,0,0],
        [1,0,0,1],
        [1,1,0,1],
        [0,1,1,1],
        [1,0,1,0]]
data = np.array(data)
model = BiMax()
model.fit(data)

# get largest bicluster
idx = np.argmax(list(model.rows_[i].sum() * model.columns_[i].sum()
                     for i in range(len(model.rows_))))
bc = np.outer(model.rows_[idx], model.columns_[idx])

# plot data and overlay largest bicluster
plt.pcolor(data, cmap = "gray_r")
plt.pcolor(bc, cmap = "gray_r", alpha=0.7)
plt.axis('scaled')
plt.xticks([1,2,3,4])
plt.yticks([1,2,3,4,5,6,7])
#plt.savefig('./images/bimax_example.png', bbox_inches='tight')
plt.show()