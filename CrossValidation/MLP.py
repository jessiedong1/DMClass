import numpy as np
np.random.seed(333)
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import regularizers
import keras_metrics

def main():
    trainset1 = np.load('train_set1.npy')
    trainset2 = np.load('train_set2.npy')
    trainset3 = np.load('train_set3.npy')
    trainset4 = np.load('train_set4.npy')
    trainset5 = np.load('train_set5.npy')

    testset1 = np.load('test_set1.npy')
    testset2 = np.load('test_set2.npy')
    testset3 = np.load('test_set3.npy')
    testset4 = np.load('test_set4.npy')
    testset5 = np.load('test_set5.npy')

    # Fold 1
    trainset1_X, trainset1_y = split_x_y(trainset1)
    testset1_X, testset1_y = split_x_y(testset1)
    val_acc1, val_pre1, val_recall1 = Cro_val(trainset1_X, trainset1_y, testset1_X, testset1_y)
    """
    result1 = MLP(trainset1_X, trainset1_y, testset1_X, testset1_y, epoch, batch_size)

    show_result(result1)
    result1 = result1.history
    result1.keys()
    val_acc1 = result1['val_acc']
    val_acc1 = np.sort(val_acc1) 
    val_acc1 = val_acc1[epoch]
    val_pre1 = result1['val_precision']
    val_pre1 = np.sort(val_pre1)
    val_pre1 = val_pre1[epoch]
    val_recall1 = result1['val_recall']
    val_recall1 = np.sort(val_recall1 )
    val_recall1  = val_recall1 [epoch]
    """
    # Fold 2
    trainset2_X, trainset2_y = split_x_y(trainset2)
    testset2_X, testset2_y = split_x_y(testset2)
    val_acc2, val_pre2, val_recall2 = Cro_val(trainset2_X, trainset2_y, testset2_X, testset2_y)

    # result2 = MLP(trainset2_X, trainset2_y, testset2_X, testset2_y, epoch, batch_size)
    # show_result(result2)

    # Fold 3
    trainset3_X, trainset3_y = split_x_y(trainset3)
    testset3_X, testset3_y = split_x_y(testset3)
    val_acc3, val_pre3, val_recall3 = Cro_val(trainset3_X, trainset3_y, testset3_X, testset3_y)
    # result3 = MLP(trainset3_X, trainset3_y, testset3_X, testset3_y, epoch, batch_size)
    # show_result(result3)

    # Fold 4
    trainset4_X, trainset4_y = split_x_y(trainset4)
    testset4_X, testset4_y = split_x_y(testset4)
    val_acc4, val_pre4, val_recall4 = Cro_val(trainset4_X, trainset4_y, testset4_X, testset4_y)
    # result4 = MLP(trainset4_X, trainset4_y, testset4_X, testset4_y, epoch, batch_size)
    # show_result(result4)

    # Fold 5
    trainset5_X, trainset5_y = split_x_y(trainset5)
    testset5_X, testset5_y = split_x_y(testset5)
    val_acc5, val_pre5, val_recall5 = Cro_val(trainset5_X, trainset5_y, testset5_X, testset5_y)
    # result5 = MLP(trainset5_X, trainset5_y, testset5_X, testset5_y, epoch, batch_size)
    # show_result(result5)

    # Show Final results
    print("Fold 1: ")
    print("Accuracy : {:.4f}".format(val_acc1))
    print("Precision: {:.4f}".format(val_pre1))
    print(" Recall: {:.4f}".format(val_recall1))
    print()

    print("Fold 2: ")
    print("Accuracy : {:.4f}".format(val_acc2))
    print("Precision: {:.4f}".format(val_pre2))
    print(" Recall: {:.4f}".format(val_recall2))
    print()

    print("Fold 3: ")
    print("Accuracy : {:.4f}".format(val_acc3))
    print("Precision: {:.4f}".format(val_pre3))
    print(" Recall: {:.4f}".format(val_recall3))
    print()
    print("Fold 4: ")
    print("Accuracy : {:.4f}".format(val_acc4))
    print("Precision: {:.4f}".format(val_pre4))
    print(" Recall: {:.4f}".format(val_recall4))
    print()
    print("Fold 5: ")
    print("Accuracy : {:.4f}".format(val_acc5))
    print("Precision: {:.4f}".format(val_pre5))
    print(" Recall: {:.4f}".format(val_recall5))
    print()
    avg_acc = float((val_acc1 + val_acc2 + val_acc3 + val_acc4 + val_acc5) / 5)
    avg_pre = float((val_pre1 + val_pre2 + val_pre3 + val_pre4 + val_pre5) / 5)
    avg_recall = float((val_recall1 + val_recall2 + val_recall3 + val_recall4 + val_recall5) / 5)

    print("Accuracy Ave: {:.4f}".format(avg_acc))
    print("Precision Ave: {:.4f}".format(avg_pre))
    print("Recall Ave: {:.4f}".format(avg_recall))
def split_x_y(dataset):
    columns = dataset.shape[1]-1
    #print(dataset)
    x = dataset[:, 7: columns]
    #print(x[:,983])
    y = dataset[:,columns]
    #print(y)


    return x,y


#Bulid MLP classifier
def MLP(train_x,train_y,test_x, test_y, epoch, batch_size):
    # One hidden layer with 60 neurons
    model = models.Sequential()

    model.add(layers.Dense(60, activation="relu", input_shape=(train_x.shape[1],),kernel_regularizer=regularizers.l2(1e-10),activity_regularizer=regularizers.l1(1e-10)))
    model.add(layers.Dropout(0.1, noise_shape=None, seed=None))

    model.add(layers.Dense(30, activation="relu", kernel_regularizer=regularizers.l2(1e-10), activity_regularizer=regularizers.l1(1e-10)))
    model.add(layers.Dense(10, activation="relu", kernel_regularizer=regularizers.l2(1e-10),activity_regularizer=regularizers.l1(1e-10)))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    #adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', metrics=['accuracy', keras_metrics.precision(),keras_metrics.recall()],optimizer='sgd')


# Get the loss and accuracy by using cross validation
    results_mlp = model.fit(train_x,train_y, epochs=epoch, batch_size= batch_size, validation_data=(test_x, test_y))
    #print(model.get_weights())
    #weights_mlp = layer.get_weights()
    return results_mlp


def Cro_val(trainset1_X, trainset1_y, testset1_X, testset1_y):
    epoch = 50
    batch_size = 30
    result1 = MLP(trainset1_X, trainset1_y, testset1_X, testset1_y, epoch, batch_size)

    show_result(result1)
    result1 = result1.history
    result1.keys()
    val_acc1 = result1['val_acc']

    val_acc1 = val_acc1[epoch]
    val_pre1 = result1['val_precision']
    #val_pre1 = np.sort(val_pre1)
    val_pre1 = val_pre1[epoch]
    val_recall1 = result1['val_recall']
    #val_recall1 = np.sort(val_recall1)
    val_recall1 = val_recall1[epoch]

    return val_acc1, val_pre1, val_recall1

def show_result(result_lp):
    result = result_lp.history
    result.keys() #dict_keys(['val_loss', 'val_acc', 'val_precision', 'val_recall', 'loss', 'acc', 'precision', 'recall'])
    #train_loss = result['loss']
    #train_loss.insert(0,0)

    #val_loss = result['val_loss']
    #val_loss.insert(0,0)
    #Accuracy
    train_acc = result['acc']
    train_acc.insert(0,0)
    val_acc = result['val_acc']
    val_acc.insert(0,0)
    epochs = range(0,len(val_acc))

    #precision
    train_pre = result['precision']
    train_pre.insert(0, 0)
    val_pre= result['val_precision']
    val_pre.insert(0, 0)

    #recall
    train_recall = result['recall']
    train_recall.insert(0, 0)
    val_recall = result['val_recall']
    val_recall.insert(0, 0)

    plt.title("MLP Classifier")
    plt.subplot(3,1,2)
    plt.plot(epochs, train_pre, 'r', label='Train Precision')
    plt.plot(epochs, val_pre, 'b', label="Test Precision")
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(3, 1, 3)

    plt.plot(epochs, train_recall, 'r', label='Train Recall')
    plt.plot(epochs, val_recall, 'b', label="Test Recall")
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True)

    plt.subplot(3, 1, 1)
    #plt.plot(epochs, train_loss, 'green', label = 'Training loss')
    #plt.plot(epochs, val_loss, 'yellow', label = "Validation loss")
    plt.plot(epochs, train_acc, 'r', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'b', label="Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #plt.legend()
    #plt.show()
    #plt.savefig()


    #plt.subplots_adjust(bottom=0.25, top=0.75)
    plt.show()




main()