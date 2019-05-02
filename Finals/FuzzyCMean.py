import numpy as np
from matplotlib import pyplot as plt

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
    #print(dis)
    clusters = Membership_dis(dis)
    New_Centers(clusters)


def Mahan_dis(center0,center1, data):
    dis0 = np.zeros((10,1),dtype=float)
    dis1 = np.zeros((10, 1), dtype=float)

    for i in range(0,10):
        dis0[i] = abs(data[i][0]-center0[0])+abs(data[i][1]-center0[1])
        dis1[i] = abs(data[i][0]-center1[0])+abs(data[i][1]-center1[1])

    dis = np.append(data, dis0, axis=1)
    dis = np.append(dis, dis1, axis=1)

    return dis

def Membership_dis(dis):
    cluster1 = np.zeros((10, 1), dtype=float)
    cluster2 = np.zeros((10, 1), dtype=float)
    for i in range(10):
       cluster1[i] = 1/(1+(dis[i][2]/dis[i][3])**2)
       cluster2[i] = 1 / (1 + (dis[i][3] / dis[i][2]) ** 2)



    clusters = np.append(dis, cluster1,axis=1)
    clusters = np.append(clusters, cluster2, axis=1)
    #clusters = clusters[np.argsort(clusters[:, 4])]
    print(clusters)
    return clusters

def New_Centers(dis):

    center1 = [0,0]
    center2 = [0,0]

    center1[0] = Cal_Center(dis[:,4],dis[:,0] )
    center1[1] = Cal_Center(dis[:,4],dis[:,1] )

    center2[0] = Cal_Center(dis[:, 5], dis[:, 0])
    center2[1] = Cal_Center(dis[:, 5], dis[:, 1])
    print("New Center 1 :", center1, "New Center 2: ", center2)


def Cal_Center(c,d):
    a = 0
    b = 0
    for i in range(10):
        a += ((c[i])**2)*(d[i])
        b += (c[i])**2


    return round(a/ b, 2)



main()