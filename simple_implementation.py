import numpy as np
from data_preparation import Dataset
from model import NN, Layer
import time
import matplotlib.pyplot as plt

def get_image(dataset, i):
    return dataset.__getitem__(i)[0][0,:,:]

def vectorize_input(dataset):
    m = dataset.__len__()
    X = np.zeros(shape=(28*28, m))

    for i in range(m):
        x = get_image(dataset, i).view(-1).numpy().reshape(28*28,1)
        X[:,i] = x[:,0]

    return X

def softmax(Z):
            shiftZ = Z - np.max(Z, axis=0, keepdims=True)
            exps = np.exp(shiftZ)
            sum_exps = np.sum(exps, axis=0, keepdims=True)
            A = exps / sum_exps
            return A

def vectorize_output(dataset):
    m = dataset.__len__()
    Y = np.zeros(shape=(10,m))
    for i in range(m):
        Y[dataset.targets[i],i] = 1

    return Y

if __name__ == "__main__":
    dataset = Dataset()
    data = dataset.train_data

    m = data.__len__()

    X = vectorize_input(data)
    Y = vectorize_output(data)

    W_1 = np.random.randn(64, 28*28) * np.sqrt(2. / 28*28)
    W_2 = np.random.randn(10, 64) * np.sqrt(2. / 64)
    b_1 = np.zeros(shape=(64,1))
    b_2 = np.zeros(shape=(10,1))
    lr = 0.01

    total_epochs = 50
    T = list(range(total_epochs))
    cost = []

    start = time.time()
    

    for epoch in range(total_epochs):
        Z_1 = W_1 @ X + b_1
        A_1 = np.maximum(Z_1,0)
        Z_2 = W_2 @ A_1 + b_2
        A_2 = softmax(Z_2)

        dZ_2 = A_2 - Y
        dW_2 = (dZ_2 @ A_1.T) / m
        db_2 = np.sum(dZ_2, axis=1, keepdims=True)
        dZ_1 = (W_2.T @ dZ_2) * np.where(Z_1 > 0, 1, 0)

        dW_1 = (dZ_1 @ X.T) / m
        db_1 = np.sum(dZ_1, axis=1, keepdims=True) / m

        W_2 -= lr * dW_2
        b_2 -= lr * db_2
        W_1 -= lr * dW_1
        b_1 -= lr * db_1

        loss = -np.mean(np.sum(Y * np.log(A_2 + 1e-9), axis=0)) 
        cost.append(loss)
        

    end = time.time()
    duration = end-start
        
    print(f"Training {total_epochs} epochs took {duration} seconds")

    plt.plot(T,cost)
    plt.show()

    test_data = dataset.test_data
    n = test_data.__len__()

    X = vectorize_input(test_data)
    Y = test_data.targets.numpy().reshape(1,n)
    
    Z_1 = W_1 @ X + b_1
    A_1 = np.maximum(Z_1,0)
    Z_2 = W_2 @ A_1 + b_2
    A_2 = softmax(Z_2)


    predictions = np.argmax(A_2, axis=0).reshape(1,n)
    accuracy = np.mean(predictions == Y)

    print("Achieved accuracy of ", accuracy * 100, "%")
