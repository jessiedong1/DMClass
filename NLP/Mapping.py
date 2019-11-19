import pandas as pd
import numpy as np
import random as rd

def main():
    cww = r'D:\Uca\Thesis\NLP\CWW.csv'
    cww = load_file(cww)
    #print(cww['CATEGORY_NAME'].unique())
    #print(cww.groupby('CATEGORY_NAME').count())

    #printCWW
    #print(cww)
    #
    # wines = r'D:\Uca\Thesis\NLP\OutputnoLabel.csv'
    # wines = load_file(wines)
    # wines['Label'] = np.where(wines['Score'] < 90, 0, 1)
    # wines = wines.drop(columns = ['Wine', 'Year', 'Score', 'Price', 'Country', 'Region', 'Issue Date'])
    #
    #
    #print(wines)
    file_Y = r'D:\Uca\Thesis\NLP\C_Y.csv'
    file_N = r'D:\Uca\Thesis\NLP\C_N.csv'
    class_Y, class_N = load_file(file_Y), load_file(file_N)
    print(class_Y)

    #Cross_Validation(class_Y, class_N, 5)



def load_file(filename):
    dataset = pd.read_csv(filename)
    dataset.head()
    dataset = pd.DataFrame(dataset)
    return dataset
def Group_Class(dataset):
    class_N, class_Y = (g for _, g in dataset.groupby('Label'))
    #class_N, class_Y = dataset.groupby('Label')
    return [class_Y, class_N]

def Cross_Validation(class_Y, class_N, n_splits):
    #num_instances = dataset.shape[0]
    #train_set = pd.DataFrame()
    #dataset_copy = dataset.copy()

    num_class_Y = class_Y.shape[0]
    num_class_N = class_N.shape[0]
    print(num_class_Y, num_class_N)
    # print("90+ wines are %d, 90- %d" %(num_class_Y, num_class_N))
    # print(4263/(4263+10086))
    # print(10086/(4263+10086))

    foldsize_Y = int(num_class_Y / n_splits)
    print(foldsize_Y)
    foldsize_N = int(num_class_N / n_splits)
    print(foldsize_N)

#Split 90+ into 5 folds
    ry = rd.sample(range(0, num_class_Y), num_class_Y)

    #print(ry1)
    ry1 = ry[0:(foldsize_Y)]
    ry2 = ry[foldsize_Y: (foldsize_Y * 2)]
    ry3 = ry[foldsize_Y * 2:(foldsize_Y * 3 )]
    ry4 = ry[foldsize_Y * 3:(foldsize_Y * 4 )]
    ry5 = ry[foldsize_Y * 4: (num_class_Y )]

    Y_1 = class_Y.iloc[ry1, :]
    Y_2 = class_Y.iloc[ry2, :]
    Y_3 = class_Y.iloc[ry3, :]
    Y_4 = class_Y.iloc[ry4, :]
    Y_5 = class_Y.iloc[ry5, :]
    """
    print(np.sort(ry1))
    print(np.sort(ry2))
    print(np.sort(ry3))
    print(np.sort(ry4))
    print(np.sort(ry5))
    """

#Split 90- into 5 folds
    rn = rd.sample(range(0, num_class_N), num_class_N)
    #rn = np.array(rn)

    rn1 = rn[0:(foldsize_N)]
    #rn1 = np.sort(rn1)
    #print(rn1)
    rn2 = rn[foldsize_N: (foldsize_N * 2 )]
    #rn2 = np.sort(rn2)
    #print(rn2)
    rn3 = rn[foldsize_N * 2:(foldsize_N * 3 )]
    rn4 = rn[foldsize_N * 3:(foldsize_N * 4 )]
    rn5 = rn[foldsize_N * 4: (num_class_N )]
    # fold_size = int(num_instances / n_splits)

    N_1 = class_N.iloc[rn1, :]
    N_2 = class_N.iloc[rn2, :]
    N_3 = class_N.iloc[rn3, :]
    N_4 = class_N.iloc[rn4, :]
    N_5 = class_N.iloc[rn5, :]

    """
    #print(np.sort(rn1))
    print(np.sort(rn1))
    #print(N_1)
    print(np.sort(rn2))
    #print(N_2)
    print(np.sort(rn3))
    print(np.sort(rn4))
    print(np.sort(rn5))
    """

#Fold 1
    train_set1 = pd.concat([Y_2,Y_3,Y_4,Y_5, N_2,N_3,N_4,N_5], ignore_index=True)
    train_set1 = train_set1
    test_set1 = pd.concat([Y_1,N_1], ignore_index=True)




#Fold 2
    train_set2 = pd.concat([Y_1,Y_3,Y_4,Y_5, N_1,N_3,N_4,N_5], ignore_index=True)
    test_set2 = pd.concat([Y_2,N_2], ignore_index=True)

    #Fold 3
    train_set3 = pd.concat([Y_1,Y_2,Y_4,Y_5, N_1,N_2,N_4,N_5], ignore_index=True)
    #print(train_set1.shape)
    test_set3 = pd.concat([Y_3,N_3], ignore_index=True)

#Fold 4
    train_set4 = pd.concat([Y_1,Y_2,Y_3,Y_5, N_1,N_2,N_3,N_5], ignore_index=True)
    #print(train_set1.shape)
    test_set4 = pd.concat([Y_4,N_4], ignore_index=True)


#Fold 5
    train_set5 = pd.concat([Y_1,Y_2,Y_3,Y_4, N_1,N_2,N_3,N_4], ignore_index=True)
    #print(train_set1.shape)
    test_set5 = pd.concat([Y_5,N_5], ignore_index=True)
    print(train_set5)

    # Save data
    # np.save('train_set1', train_set1)
    # np.save('test_set1', test_set1)
    # np.save('train_set2', train_set2)
    # np.save('test_set2', test_set2)
    # np.save('train_set3', train_set3)
    # np.save('test_set3', test_set3)
    # np.save('train_set4', train_set4)
    # np.save('test_set4', test_set4)
    # np.save('train_set5', train_set5)
    # np.save('test_set5', test_set5)

main()