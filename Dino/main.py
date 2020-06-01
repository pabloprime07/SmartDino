from game import gameplay
from game import introscreen
from generation import Generation
import numpy as np

def main():
    n_gen = 200
    isGameQuit = introscreen()
    G = Generation()
    highestscore = 0
    for i in range(n_gen):
        # isGameQuit = introscreen()
        print("iter ")
        gameplay(G,0,0)
        print("Generation :"+str(i))
        
        print("Highscore :"+str([x.fitness for x in G.bois]))
        print("Average Score :"+str(np.average([x.fitness for x in G.bois])))
        G.findGoodBois()
        print(G.goodBois[0].fitness)
        if G.goodBois[0].fitness > highestscore:
            highestscore = G.goodBois[0].fitness
        print(G.goodBois[0].W1,G.goodBois[0].W2)

        G.mutations()
    print("HS : ")
    print(highestscore)
        


main()
