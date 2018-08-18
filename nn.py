import numpy as np

df_nn = np.loadtxt('train_100k.csv', delimiter=",", skiprows=1)
df_truth = np.loadtxt('train_100k.truth.csv', delimiter=",", skiprows=1)

x = df_nn[:, 1:22]
y = df_truth[:, 1:3]

#units of scale
x = x/np.amax(x, axis=0) #maximum in array x
y = y/np.amax(y, axis=0) #maximum in array y

class Neural_Network(object):
    def __init__ (self):
        #parameters
        self.inputSize = 20
        self.outputSize = 2
        self.hiddenSize = 1
        #weights
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) #input weight matrix for the hidden layer
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) #matrix from the hidden layer to the output layer

    def forward (self, x):
        #spread through our network
        self.z = np.dot(x, self.w1) #product point of X (input) and first set of weights
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.w2) #product of the hidden layer (z2) and second set of weights
        o = self.sigmoid(self.z3) #final activation function
        return o

    def sigmoid (self, s):
        #activation function
        return 1/(1 + np.exp(-s))

    def sigmoidPrime (self, s):
        #sigmoid derivative
        return s * (1 - s)

    def backward (self, x, y, o):
        # propagation back through the network
        self.o_error = y - o #exit error
        self.o_delta = self.o_error * self.sigmoidPrime(o) #applying sigmoid derivative to error
        self.z2_error = self.o_delta.dot(self.w2.T) #z2_error: how much our hidden layer weights contributed to the output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) #applying sigmoid derivative in z2_error
        self.w1 += x.T.dot(self.z2_delta) #adjusting the first set (input -> hidden) weights
        self.w2 += self.z2.T.dot(self.o_delta) #

    def train (self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)  

    def predict (self):
        print("Predicted data based on weights trained: ")
        print("Input (scale):   \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))


NN = Neural_Network()

for i in range(1000): #trains NN 1,000 times
    print("Input: \n" + str(x))
    print("Real Output: \n" + str(y))
    print("Expected Output: \n" + str(NN.forward(x)))
    print("Lost: \n" + str(np.mean(np.square(y - NN.forward(x))))) #soma  a media quadrada da perda
    print("\n")
    NN.train(x, y)

pred = np.loadtxt('test_100k.csv', delimiter=",", skiprows=1)    

xPredicted = pred[:, 1:22]

xPredicted = xPredicted/np.amax(xPredicted, axis=0) #maximum of our test

NN.predict()
