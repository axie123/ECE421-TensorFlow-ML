import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(3500,784)
validData = validData.reshape(100,784)
testData = testData.reshape(145,784)


def MSE(W, b, x, y, reg):
    # Your implementation here
    x = np.insert(x, 0, 1, axis=1)
    W_b = np.insert(W, 0, b)
    y = y.reshape(y.shape[0])
    MSE_Loss = (np.linalg.norm(np.dot(x, W_b) - y) ** 2) / len(x) + (reg / 2) * np.linalg.norm(W) ** 2
    return MSE_Loss

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    x = np.insert(x, 0, 1, axis=1)
    W_b = np.insert(W, 0, b)
    y = y.reshape(y.shape[0])
    grad_wb = (2 * np.dot(np.transpose(x), (np.dot(x, W_b) - y))) / len(x) + reg * W_b
    return grad_wb[1:], grad_wb[0]

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    x = np.insert(x, 0, 1, axis=1)
    W_b = np.insert(W, 0, b)
    y = y.reshape(y.shape[0])
    z = np.dot(x,W_b)
    sigmoid = 1/(1 + np.exp(-z))
    CELoss = np.linalg.norm(np.dot(-y, np.log(sigmoid)) - np.dot(1 - y, np.log(1-sigmoid)))/len(x) + (reg/2)*np.linalg.norm(W)**2
    return CELoss

def gradCE(W, b, x, y, reg):
    # Your implementation here
    x = np.insert(x, 0, 1, axis=1)
    W_b = np.insert(W, 0, b)
    y = y.reshape(y.shape[0])
    z = np.dot(x, W_b)
    sigmoid = 1 / (1 + np.exp(-z))
    error = sigmoid - y
    dL_dwb = np.dot(np.transpose(x), error) / len(x)
    return dL_dwb[1:], dL_dwb[0]

def accuracy(W, b, x, y):
   x = np.insert(x, 0, 1, axis=1)
   W_b = np.insert(W, 0, b)
   y = y.reshape(y.shape[0])
   pred = np.dot(x,W_b)
   pred = np.where(pred >= 0.5, 1, 0)
   accuracy = np.sum(pred == y)/len(pred)
   return accuracy

def plotting_loss(epoch, training_error, validation_error, testing_error, title):
    epoch_idx = np.arange(0, epoch)
    plt.figure(figsize=(10,10))
    plt.plot(epoch_idx,training_error)
    plt.plot(epoch_idx,validation_error)
    plt.plot(epoch_idx,testing_error)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss', 'Testing Loss'])
    plt.title(title)
    plt.show()

def plotting_accuracy(epoch, training_accuracy, validation_accuracy, testing_accuracy, title):
    epoch_idx = np.arange(0, epoch)
    plt.figure(figsize=(10,10))
    plt.plot(epoch_idx,training_accuracy)
    plt.plot(epoch_idx,validation_accuracy)
    plt.plot(epoch_idx,testing_accuracy)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'])
    plt.title(title)
    plt.show()

W_i = np.random.normal(0.5,0.5,28*28) #np.zeros(28*28)
b_i = 0
l_r = [0.005, 0.001, 0.0001]
reg = 0 #[0.001, 0.1, 0.5]
epochs = 5000
error_tol = 10**-7

weight = np.empty([len(l_r), 28*28])
bias = np.empty([len(l_r), 1])
training_error = np.empty([len(l_r), epochs])
training_accuracy = np.empty([len(l_r), epochs])
validation_error = np.empty([len(l_r), epochs])
validation_accuracy = np.empty([len(l_r), epochs])
testing_error = np.empty([len(l_r), epochs])
testing_accuracy = np.empty([len(l_r), epochs])

def grad_descent(W, b, x, y, v_x, v_y, test_x, test_y, alpha, epochs, reg, error_tol, lossType = 'MSE'):
    # Your implementation here
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    test_loss = []
    test_accuracy = []
    for i in range(epochs):
        if lossType == 'MSE':
            dl_dw, dl_db = gradMSE(W, b, x, y, reg)  # The gradient based on loss for each image
        elif lossType == 'CE':
            dl_dw, dl_db = gradCE(W, b, x, y, reg)  # The gradient based on loss for each image
        W_new = W - alpha * (dl_dw)  # Updates weights
        b -= alpha * (dl_db)  # Updates bias

        if (np.linalg.norm(W_new - W) < error_tol):
            return [W, b, training_loss, validation_loss, test_loss, training_accuracy, validation_accuracy,
                    test_accuracy]

        W = W_new

        # Training Loss:
        t_accuracy = accuracy(W, b, x, y)
        if lossType == 'MSE':
            t_loss = MSE(W, b, x, y, reg)
        elif lossType == 'CE':
            t_loss = crossEntropyLoss(W, b, x, y, reg)
        print("Epoch: %d, Training Loss: %0.2f, Training Accuracy: %0.2f" % (i, t_loss, t_accuracy))
        training_loss += [t_loss]
        training_accuracy += [t_accuracy]

        # Validation Loss:
        v_accuracy = accuracy(W, b, v_x, v_y)
        if lossType == 'MSE':
            v_loss = MSE(W, b, x, y, reg)
        elif lossType == 'CE':
            v_loss = crossEntropyLoss(W, b, x, y, reg)
        print("Epoch: %d, Validation Loss: %0.2f, Validation Accuracy: %0.2f" % (i, v_loss, v_accuracy))
        validation_loss += [v_loss]
        validation_accuracy += [v_accuracy]

        # Testing Loss:
        test_acc = accuracy(W, b, test_x, test_y)
        if lossType == 'MSE':
            te_loss = MSE(W, b, x, y, reg)
        elif lossType == 'CE':
            te_loss = crossEntropyLoss(W, b, x, y, reg)
        print("Epoch: %d, Testing Loss: %0.2f, Testing Accuracy: %0.2f" % (i, te_loss, test_acc))
        test_loss += [te_loss]
        test_accuracy += [test_acc]

    return [W, b, training_loss, validation_loss, test_loss, training_accuracy, validation_accuracy, test_accuracy]

W = W_i
b = b_i
r = 0
[weight[r], bias[r], training_error[r], validation_error[r], testing_error[r], training_accuracy[r], validation_accuracy[r], testing_accuracy[r]] = grad_descent(W, b,
                                                                                                                                                                 trainData, trainTarget,
                                                                                                                                                                 validData, validTarget,
                                                                                                                                                                 testData, testTarget,
                                                                                                                                                                 0.005, epochs, reg,
                                                                                                                                                                 error_tol,lossType = 'MSE')

epochs = 700
batch_size = 500
l_r = 0.001
b1 = None
b2 = None
e = 10e-4

def accuracySGD(predictions, labels):
    return (np.sum((predictions>=0.5)==labels) / np.shape(predictions)[0])

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    num_batches = int(3500 / batch_size)
    lbda = 0
    graph = tf.Graph()

    with graph.as_default():
        W = tf.truncated_normal(shape=(784, 1), mean=0.0, stddev=0.5, dtype=tf.float32)
        W = tf.Variable(W)
        b = tf.zeros(1)
        b = tf.Variable(b)

        x = tf.placeholder(tf.float32, shape=(batch_size, 784))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1))

        v_x = tf.placeholder(tf.float32, shape=(len(validData), 784))
        v_y = tf.placeholder(tf.int8, shape=(len(validTarget), 1))

        t_x = tf.placeholder(tf.float32, shape=(len(testData), 784))
        t_y = tf.placeholder(tf.int8, shape=(len(testTarget), 1))

        tf.set_random_seed(421)

        if lossType == "MSE":
            z = tf.matmul(x, W) + b
            loss = tf.losses.mean_squared_error(y, z)
            reg = tf.nn.l2_loss(W)
            loss += (lbda / 2.0) * reg

            v_z = tf.matmul(v_x, W) + b
            vloss = tf.losses.mean_squared_error(v_y, v_z)
            vloss += (lbda / 2.0) * reg

            t_z = tf.matmul(t_x, W) + b
            tloss = tf.losses.mean_squared_error(t_y, t_z)
            tloss += (lbda / 2.0) * reg

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=10e-4).minimize(loss)

        elif lossType == "CE":
            z = tf.sigmoid(tf.matmul(x, W) + b)
            loss = tf.losses.sigmoid_cross_entropy(y, z)
            reg = tf.nn.l2_loss(W)
            loss += (lbda / 2.0) * reg

            v_z = tf.sigmoid(tf.matmul(v_x, W) + b)
            vloss = tf.losses.sigmoid_cross_entropy(v_y, v_z)
            vloss += (lbda / 2.0) * reg

            t_z = tf.sigmoid(tf.matmul(t_x, W) + b)
            tloss = tf.losses.sigmoid_cross_entropy(t_y, t_z)
            tloss += (lbda / 2.0) * reg

            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.99).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            training_loss = []
            validation_loss = []
            testing_loss = []
            training_accuracy = []
            validation_accuracy = []
            testing_accuracy = []
            for epoch in range(epochs):
                total_loss = 0
                for n in range(num_batches):
                    x_batch = trainData[n * batch_size:(n + 1) * batch_size, ]
                    y_batch = trainTarget[n * batch_size:(n + 1) * batch_size, ]
                    _, opt_W, opt_b, train_loss, pred, v_loss, v_pred, t_loss, t_pred = session.run(
                        [optimizer, W, b, loss, z, vloss, v_z, tloss, t_z],
                        {x: x_batch,
                         y: y_batch,
                         v_x: validData,
                         v_y: validTarget,
                         t_x: testData,
                         t_y: testTarget})
                if (n % 1 == 0):
                    training_loss += [train_loss]
                    t_accuracy = accuracy(pred, y_batch)
                    training_accuracy += [t_accuracy]
                    validation_loss += [v_loss]
                    v_accuracy = accuracy(v_pred, validTarget)
                    validation_accuracy += [v_accuracy]
                    testing_loss += [t_loss]
                    t_accuracy = accuracy(t_pred, testTarget)
                    testing_accuracy += [t_accuracy]

                    print('Epoch: {}, Training Loss: {}, Training Accuracy: {}'.format(epoch, train_loss, t_accuracy))
                    print('Epoch: {}, Validation Loss: {}, Validation Accuracy: {}'.format(epoch, v_loss, v_accuracy))
                    print('Epoch: {}, Testing Loss: {}, Testing Accuracy: {}'.format(epoch, t_loss, t_accuracy))

    # Your implementation here
    return opt_W, opt_b, (pred >= 0.5), trainTarget, train_loss, optimizer, reg, training_loss, training_accuracy, validation_loss, validation_accuracy, testing_loss, testing_accuracy

opt_W, opt_b, pred, trainTarget, train_loss, optimizer, reg, training_loss, training_accuracy, validation_loss, validation_accuracy, testing_loss, testing_accuracy = buildGraph(lossType = "MSE")

plotting_loss(epochs, training_loss, validation_loss, testing_loss,"BCE Losses of SGD w/ epsilon of 1e-04, lr of 0.001, and batch of 500 on notMNIST")
plotting_accuracy(epochs, training_accuracy, validation_accuracy, testing_accuracy, "BCE Accuracy of SGD w/ epsilon of 1e-04, lr of 0.001, and batch of 500 on notMNIST")

# Plots the graph for comparisons:
opt_W, opt_b, pred, trainTarget, train_loss, optimizer, reg, ce_training_loss, ce_training_accuracy, ce_validation_loss, ce_validation_accuracy, ce_testing_loss, ce_testing_accuracy = buildGraph(lossType = "CE")

def plotting_accuracy_compare(epoch, training_accuracy, validation_accuracy, testing_accuracy, title):
    epoch_idx = np.arange(0, epoch)
    plt.figure(figsize=(10,10))
    plt.plot(epoch_idx,training_accuracy)
    plt.plot(epoch_idx,validation_accuracy)
    plt.plot(epoch_idx,testing_accuracy)
    plt.plot(epoch_idx,ce_training_accuracy)
    plt.plot(epoch_idx,ce_validation_accuracy)
    plt.plot(epoch_idx,ce_testing_accuracy)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy BGD', 'Validation Accuracy BGD', 'Testing Accuracy BGD','Training Accuracy SGD', 'Validation Accuracy SGD', 'Testing Accuracy SGD'])
    plt.title(title)
    plt.show()