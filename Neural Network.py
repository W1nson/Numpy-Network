import numpy as np 
import cv2 as cv 
from matplotlib import pyplot as plt

np.random.seed(0)

class model: 
    
    def __init__(self):
        self.layer = []

    def add(self, layers):
        self.layer.append(layers)

    def train(self, train_in, train_out, count): #test_input, test_out, lr):
        self.layer.insert(0, layers(train_in))
        print(self.layer[0].data)
        for i in range(1, len(self.layer)):
            self.layer[i].forward(self.layer[i-1])
            print('before:', self.layer[i].weight)
        
        self.layer[len(self.layer)-1].backpropagation(train_out , count)
            
        for i in range(len(self.layer)): 
            print('after:', self.layer[i].weight)

class layers: 
    
    def __init__(self, train_in = np.zeros(shape=(0)), num_input = 0, num_output = 0, actua = '', bias = 0): 
        self.num_input = num_input 
        self.num_output = num_output 
        self.data = np.zeros(shape=(0))
        self.weight = np.zeros(shape=(num_input, num_output))
        if num_input != 0 and num_output != 0: 
            self.weight = np.random.random_sample((num_input, num_output))
        self.actua = actua
        self.derive = 0
        self.output = train_in
        self.bias = bias
        self.count = 1 
        
    def actuation(self): 
        if self.actua == 'sigmoid': 
            self.output = np.where(True, 1/(1+np.exp(-(self.data))),0)
            self.derive = self.output*(1-self.output)

        elif self.actua == 'relu': 
            self.output = np.where(self.data >= 0.0, self.data, 0)
            self.derive = np.where(self.output > 0.0, self.derive, 1)
            self.derive = np.where(self.output < 0.0, self.derive, 0)

        elif self.actua == 'leakyrelu': 
            self.output = np.where(self.data >= 0.0, self.data, self.data * 0.01)
            self.derive = np.where(self.output > 0.0, self.derive, 1)
            self.derive = np.where(self.output < 0.0, self.derive, 0.1)


    def forward(self, layer): 
        if not isinstance(layer, layers):
            self.data = layer.dot(self.weight) + self.bias
            self.actuation()
            return self.output
        else: 
            self.data = np.array(layer.output).dot(self.weight) + self.bias
            self.actuation()
            return self.output

    def backpropagation(self, target, count):
         #error detection
        self.error = (1/2) * np.square(target - self.data)
        self.totalError = np.sum(self.error)
             #backpropagation:
             # d(Etotal)/d(weight) = d(Etotal)/d(out) * d(out)/d(data) * d(data)/d(weight)
        self.adj = (self.output - target) * (self.output*(1-self.output)) * self.data 
      



    def size(self): 
        return np.array[self.num_input, self.num_output]
    
    
#data import and clean     
img = cv.imread('digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape[0])
print(img.shape[1])

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells) 
#20X20 pixels per image
plt.imshow(x[5][0])
print(x[5][0])


network = model() 
network.add(layers(0 ,20, 100, 'relu'))
network.add(layers(0 ,100, 50, 'relu')) 
network.add(layers(0 ,50, 10, 'relu'))





