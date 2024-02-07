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

    model = NN([
        Layer(28*28, 512, "relu"),
        Layer(512, 512, "relu"),
        Layer(512, 10, "softmax")
    ], lr = 1e-3)

    total_epochs = 10
    T = list(range(total_epochs))
    cost = []

    start = time.time()

    for epoch in range(total_epochs):
        print("epoch: ", epoch)
        for batch in range(m // 64):
            input_batch = X[:, batch * 64 : min(m, (batch+1) * 64)]
            output_batch = Y[:, batch * 64 : min(m, (batch+1) * 64)]
            model.forward(input_batch)
            model.backward(input_batch,output_batch)
        cost.append(model.loss(X,Y))
        

    end = time.time()
    duration = end-start
        
    print(f"Training {total_epochs} epochs took {duration} seconds")

    plt.plot(T,cost)
    plt.show()

    test_data = dataset.test_data
    n = test_data.__len__()

    X = vectorize_input(test_data)
    Y = test_data.targets.numpy().reshape(1,n)
    A = model.forward(X)


    predictions = np.argmax(A, axis=0).reshape(1,n)
    accuracy = np.mean(predictions == Y)

    print("Achieved accuracy of ", accuracy * 100, "%")
