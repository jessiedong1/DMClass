from NavieBayes.MLP import *

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


main()