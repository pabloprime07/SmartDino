import numpy as np

np.random.seed(70)

class MySmartDino:

    def __init__(self):

        self.input_size = 5
        self.hidden_size = 10
        # self.hidden_size2 = 7
        self.output_size = 2
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        # self.W1 = np.array([[ 1.06104707,-1.46405875,-1.50112142, 0.04454776, -0.55979429], [-0.94486506,  1.92510766 ,-0.80251648, -2.00174788, -0.19912911], [-0.0781342,   1.18870985, -0.1125834,  -1.27053368, -0.03725404], [-0.4614089,   0.12624019, -1.69853122 , 0.87368621,  1.89923724]]) 
        # self.W2 = np.array([[-0.27458325], [-1.08315967], [-0.76981992], [-0.34408659], [-0.58388135]])
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        # self.W3 = np.random.randn(self.hidden_size2, )
        #self.W1 = np.array([[0.27669465,-0.93263789,-0.38175547,0.55580387,0.47993602],[-0.73937507,3.29347054,-3.480049,-2.088545,0.57429839],[0.45700636,0.91241193,-1.51604491,-1.53303842,-1.8718016],[0.82702108,4.7366557,-1.56204706,-0.42656146,0.18151335]]) 
        #self.W2 = np.array([[-2.55698851],[0.23018271],[-2.32387562],[0.90671722],[-2.40331313]])
        # self.W1 = self.W1 + np.random.normal(0,1,(self.input_size, self.hidden_size))
        # self.W2 = self.W2 + np.random.normal(0,1,(self.hidden_size, self.output_size))
        #448
        self.fitness = 0
    
    def setparam(self, w1, w2):
        self.W1 = np.array(w1)
        self.W2 = np.array(w2)

    def predict(self, inputs):
        # print(inputs)
        # print(self.W1,self.W2)
        z2 = inputs@self.W1
        # print(z2)
        a2 = np.tanh(z2)
        # print(a2)        
        z3 = a2@self.W2
        # print(z3)
        yHat = np.tanh(z3)
        # print(yHat)
        # return yHat
        return 1 / (1 + np.exp(-yHat))

    # def updatefitness(self, fitness):
    #     self.fitness = fitness


