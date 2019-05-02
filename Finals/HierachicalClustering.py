import numpy as np
from matplotlib import pyplot as plt
import matplotlib

colors = ['r', 'b']
def main():
    # data = [[0,10],[1,3],[2,3],
    #         [2,8],[5,1],[5,6],
    #         [7,3],[7,4],[8,5],[9,6]]
    data = {0: [0, 10], 1: [1, 3], 2: [2, 3],
            3: [2, 8], 4:[5, 1], 5:[5, 6],
            6:[7, 3],7: [7, 4], 8:[8, 5], 9:[9, 6]}

    dis_all = Mahan_dis_All(data)
    #print(dis_all)
    #print(data)
    #draw_graph(data)
    data[19] = Cal_Average(data[1], data[2])
    data[49] = Cal_Average(data[6], data[7])
    data[77] = Cal_Average(data[8], data[9])
    data[103] = Cal_Average(data[49],data[77])
    data[104] = Cal_Average(data[0], data[3])
    data[124] = Cal_Average(data[5], data[103])
    data[129] = Cal_Average(data[4], data[19])
    data[136] = Cal_Average(data[124], data[129])
    print(data)
    data1 = data.copy()
    data1.pop(1)
    data1.pop(2)
    data1.pop(6)
    data1.pop(7)
    data1.pop(8)
    data1.pop(9)
    data1.pop(49)
    data1.pop(77)
    data1.pop(0)
    data1.pop(3)
    data1.pop(5)
    data1.pop(103)
    data1.pop(4)
    data1.pop(19)
    data1.pop(124)
    data1.pop(129)
    #print(data1.keys())
    Mahan_dis_ALL_1(data1)



    draw_graph(data)

def Mahan_dis1(point0, point1):
    a =  abs(point0[0]-point1[0])+abs(point0[1]-point1[1])
    return a

def Mahan_dis_All(data):
    #A(10,2)
    dis_all = np.zeros((45,4), dtype=float)
    a =10
    b = 0
    for key in data:
        b = b+1
        for j in range(b,10):
            dis_all[a-10][0] = a
            dis_all[a-10][1] = key
            dis_all[a-10][2] = j
            dis_all[a-10][3] = Mahan_dis1(data[key],data[j])
            a = a + 1

    dis_all = (dis_all[dis_all[:,3].argsort(kind='mergesort')])
    #print(dis_all)
    return dis_all

def Mahan_dis_ALL_1(data):
    #A(8,2)=28
    #A(7,2) = 21
    #A(6,2) = 15
    #A(5,2) = 10
    #A(4,2) = 6
    #A(3,2) = 3
    #A(2,2) =1
    dis_all = np.zeros((1,4), dtype=float)
    #start with 55
    a = 137
    for key in data:
        for j in data:
            if(key < j):
                dis_all[a - 137][0] = a
                dis_all[a - 137][1] = key
                dis_all[a - 137][2] = j
                dis_all[a - 137][3] = Mahan_dis1(data[key], data[j])
                a = a + 1

    dis_all = (dis_all[dis_all[:,3].argsort(kind='mergesort')])
    print(dis_all)
    return dis_all
def Cal_Average(point1, point2):
    a = 0
    b = 0
    a = round((point1[0]+point2[0])/2,2)
    b = round((point1[1]+point2[1])/2,2)

    return [a,b]


def draw_graph(data):

    new_data = {"x": [], "y": [], "label": []}
    for key, value in data.items():
        new_data["x"].append(value[0])
        new_data["y"].append(value[1])
        new_data["label"].append(key)

    # display scatter plot data
    #plt.title('Initial Plot', fontsize=20)
    plt.scatter(new_data["x"], new_data["y"], marker='o')

    # add labels
    for label, x, y in zip(new_data["label"], new_data["x"], new_data["y"]):
        plt.annotate(label, xy=(x, y))
    x1, y1 = data[1], data[2]
    x2,y2 = data[6], data[7]
    x3,y3 = data[8],data[9]
    x4,y4 = data[7], data[8]
    x5,y5 = data[0],data[3]
    x6,y6 = data[5], data[7]
    x7, y7 = data[2], data[4]
    x8, y8 = data[2], data[5]
    x9,y9 = data[3], data[5]
    plt.plot([x1[0], y1[0]],[x1[1], y1[1]],marker = 'o')
    plt.plot([x2[0], y2[0]], [x2[1], y2[1]], marker='o')
    plt.plot([x3[0], y3[0]], [x3[1], y3[1]], marker='o')
    plt.plot([x4[0], y4[0]], [x4[1], y4[1]], marker='o')
    plt.plot([x5[0], y5[0]], [x5[1], y5[1]], marker='o')
    plt.plot([x6[0], y6[0]], [x6[1], y6[1]], marker='o')
    plt.plot([x7[0], y7[0]], [x7[1], y7[1]], marker='o')
    plt.plot([x8[0], y8[0]], [x8[1], y8[1]], marker='o')
    plt.plot([x9[0], y9[0]], [x9[1], y9[1]], marker='o')

    plt.show()


main()