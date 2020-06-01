from percp import MySmartDino
import numpy as np
import copy

np.random.seed(70)
genSize = 100

class Generation:

    def __init__(self):
        self.bois = list()
        self.goodBois = list()
        self.studbois = list()
        self.max_fit = list()
        for _ in range(genSize):
            self.bois.append(MySmartDino())

    def findGoodBois(self):
        self.bois.sort(key=lambda x: x.fitness, reverse=True)
        self.bois = self.bois[:int(genSize/10)]
        self.goodBois = self.bois[:]
        l = self.goodBois + self.studbois
        l.sort(key=lambda x: x.fitness, reverse=True)
        l = l[:int(genSize/16)]
        self.studbois = l
    
    def mutations(self):
        # self.bois = list()
        
        l = self.goodBois
        l += self.studbois[:]
        # print(l)
        # print(self.goodBois)
        while len(self.bois) < genSize:
            boi1 = np.random.choice(l)
            boi2 = np.random.choice(l)
            self.bois.append(self.mutate(self.cross_over(boi1, boi2)))
        
        # while len(self.bois) < genSize:
        # boi = self.studbois[0]
        # self.bois.append(self.mutate(boi))
        # boi = self.studbois[1]
        # self.bois.append(self.mutate(boi))
        # while len(self.bois) < genSize:
        #     boi = np.random.choice(self.goodBois)
        #     self.bois.append(self.mutate(boi))

    def cross_over(self, boi1, boi2):
        new_boi = copy.deepcopy(boi1)
        other_boi = copy.deepcopy(boi2)
        cut_location = int(len(new_boi.W1) * np.random.uniform(0, 1))
        for i in range(cut_location):
            new_boi.W1[i], other_boi.W1[i] = other_boi.W1[i], new_boi.W1[i]
        cut_location = int(len(new_boi.W2) * np.random.uniform(0, 1))
        for i in range(cut_location):
            new_boi.W2[i], other_boi.W2[i] = other_boi.W2[i], new_boi.W2[i]
        # new_boi = copy.deepcopy(boi1)
        # new_boi.W1 = (boi1.W1+boi2.W1)/2
        # new_boi.W2 = (boi1.W2+boi2.W2)/2
        return new_boi

    def __mutate_weights(self, weights):
        if np.random.uniform(0, 1) < 0.2:
            return weights + weights*(np.random.normal(0, 0.8))
        else:
            return 0

    def mutate(self, boi):
        new_boi = copy.deepcopy(boi)
        new_boi.W1 += self.__mutate_weights(new_boi.W1)
        new_boi.W2 += self.__mutate_weights(new_boi.W2)
        return new_boi
    
        
