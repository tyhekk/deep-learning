import loadMNIST
import numpy as np
import random
import pickle

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]


    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data = None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            if test_data:
                print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complte".format(j))

    def update_mini_batch(self,mini_batch,eta):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nable_b,delta_nable_w =  self.backprop(x,y)
            #print(len(nable_b[1]))
            #print(len(delta_nable_b[1]))
            nable_b = [nb + dnb for nb,dnb in zip(nable_b,delta_nable_b)]
            nable_w = [nw + dnw for nw,dnw in zip(nable_w,delta_nable_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nable_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nable_b)]


    def backprop(self,x,y):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]

        #print(self.biases[0].shape)

        activation = x
        activations = [x]

        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            
            zs.append(z)
            activation = sigmoid(z)
            
            activations.append(activation)
            
        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])

        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta,activations[-2].transpose())
     
        for k in range(2,self.num_layers):
            z = zs[-k]
  
            sp = sigmoid_prime(z)
  
            delta = np.dot(self.weights[-k+1].transpose(),delta)*sp
            nable_b[-k] = delta

            nable_w[-k] = np.dot(delta,activations[-k-1].transpose())

        return (nable_b,nable_w)

    def evaluate(self,test_data):
        print((np.argmax(self.feedforward(test_data[0][0])),test_data[0][1]))
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]

        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self,output_activations,y):
        output_activations[y] -= 1
        return (output_activations)

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
        return sigmoid(z)*(1 - sigmoid(z))


if __name__ == '__main__':
    if 0:
        train_label = loadMNIST.getTrainLabel('./Data/train-labels-idx1-ubyte.gz')
        train_data = loadMNIST.getTrainData('./Data/train-images-idx3-ubyte.gz')
        test_label = loadMNIST.getTestLabel('./Data/t10k-labels-idx1-ubyte.gz')
        test_data = loadMNIST.getTestData('./Data/t10k-images-idx3-ubyte.gz')

        train_datas = [((np.array(data)/255).reshape(len(data),1),label) for data,label in zip(train_data,train_label)];
        test_datas = [((np.array(data)/255).reshape(len(data),1),label) for data,label in zip(test_data,test_label)];

        with open('./train_data.pkl','wb') as train_file:
            pickle.dump(train_datas,train_file)

        with open('./test_data.pkl','wb') as test_file:
            pickle.dump(test_datas,test_file)

    with open('./train_data.pkl','rb') as f:
        train_data = pickle.load(f)
    with open('./test_data.pkl','rb') as f:
        test_data = pickle.load(f)

    net = Network([784,60,10])
    net.SGD(train_data,30,10,3.0,test_data = test_data)
    
    




    













            
        
                        
