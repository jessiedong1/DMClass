import numpy as np
from matplotlib import pyplot as plt
import matplotlib

colors = ['r', 'b']
def main():
    data = [[0,10],[1,3],[2,3],
            [2,8],[5,1],[5,6],
            [7,3],[7,4],[8,5],[9,6]]
    data = np.array(data)

    # Plot the data
    x, y = data.T
    plt.scatter(x,y)
    #plt.show()
    center1 = [5,1]
    center2 = [5,6]
    print("Center 1: [5 1], Center 2: [5 6]")
    dis = Mahan_dis(center1,center2,data)
    clusters = Compare_dis(dis)
    new_center0, new_center1 = show_plot(clusters,center1,center2)

    dis = Mahan_dis(new_center0, new_center1, data)
    clusters = Compare_dis(dis)
    new_center01, new_center11 = show_plot(clusters, new_center0, new_center1)

    dis = Mahan_dis(new_center01, new_center11, data)
    clusters = Compare_dis(dis)
    new_center011, new_center111 = show_plot(clusters,new_center01, new_center11)


def Mahan_dis(center0,center1, data):
    dis0 = np.zeros((10,1),dtype=float)
    dis1 = np.zeros((10, 1), dtype=float)

    for i in range(0,10):
        dis0[i] = abs(data[i][0]-center0[0])+abs(data[i][1]-center0[1])
        dis1[i] = abs(data[i][0]-center1[0])+abs(data[i][1]-center1[1])

    dis = np.append(data, dis0, axis=1)
    dis = np.append(dis, dis1, axis=1)

    return dis

def Compare_dis(dis):
    clusters = np.zeros((10, 1), dtype=int)
    for i in range(10):
        if(dis[i][2] > dis[i][3]):
            clusters[i] = 1


    clusters = np.append(dis, clusters,axis=1)
    #clusters = clusters[np.argsort(clusters[:, 4])]

    return clusters

def show_plot(clusters,center1,center2):
    print(clusters)
    X = clusters[:,0]
    y = clusters[:,1]
    labels = clusters[:,4]
    new_center0=[0,0]
    num_class0 =0
    new_center1 = [0,0]
    num_class1 = 0
    for i in range(10):
        if(clusters[i][4] == 0):
            new_center0[0] += clusters[i][0]
            new_center0[1] += clusters[i][1]
            num_class0 = num_class0+1
        else:
            new_center1[0] += clusters[i][0]
            new_center1[1] += clusters[i][1]
            num_class1 = num_class1+1
    # print(new_center0)
    # print(new_center1)
    new_center0 = np.true_divide(new_center0,num_class0)
    new_center1 = np.true_divide(new_center1, num_class1)
    print()
    print("Center 1: :", new_center0, " Center 2: ", new_center1)

    #print(new_center0)
    #print(new_center1)

    plt.scatter(X, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.scatter(center1[0], center1[1], marker="x", color='r')
    plt.scatter(center2[0], center2[1], marker="x", color='b')
    #plt.plot(std, std + 0, linestyle='solid')
    plt.show()

    return new_center0,new_center1


main()