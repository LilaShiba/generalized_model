import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt


def entropy(arr):
    e,p = [],[]
    for delta in arr:
        u, counts = np.unique(delta, return_counts=True)
        probabilities = counts/np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        entropy = round(entropy,2)
        e.append(entropy)
        p.append(probabilities)
    print(p,'p')
    print(e,'e')
    return e,p


def entropyTest(tests):
    for delta in tests:
        delta = sorted(delta, reverse= False)
        n = len(delta)#len(np.unique(delta))
        h = collections.defaultdict(int)
        
        for node in delta:
            h[node] += 1
        p = []
        for x in h.keys():
           p.append((h[x] / n))
        print(p)
        entropy = round(-np.sum(p * np.log2(p)),2)
        print(entropy)

        #Plot the PDF
        plt.plot(h.keys(),p,'o')
        plt.ylabel('Probability')
        plt.xlabel('X')

        plt.title('PDF')
        plt.show()

    return entropy,p
t1 = np.random.rand(10)
# Array with a low entropy score
t2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
# Array with a high entropy score
t3 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# rand Distrubution
t4 = np.random.randint(0,10,1000)
# norm distrubution
mean = 0
std = 3
size = 100
t5 = np.random.normal(mean, std, size)
#plt.hist(t5)
#plt.show()
t5 = [round(x,0) for x in t5]
tests = [t5]
#entropy(tests)
entropyTest(tests)
#entropy([t5])



