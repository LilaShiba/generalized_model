import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class node():
    def __init__(self,vector):
        self.vector = np.array(vector)
        self.n = len(vector)
        self.vector_mu = np.mean(vector)
        self.distro = np.hist(self.vector)
        # pre-process vector
        self.pdf()
        self.get_entropy()
        self.get_variance()
        self.norm_vector()

    def get_corr(self,y):
        
        x = self.vector
        m = len(y)
        x_mu = np.mean(x)
        y_mu = np.mean(y)

        if self.n!=m:
            print('woof, cols not equal length')
            return 
        
        top_term = 0
        btm_term_x = 0
        btm_term_y = 0
        for i in range(n):
            top_term += (x[i] - x_mu) * (y[i] - y_mu)
            btm_term_x += (x[i] - x_mu)**2
            btm_term_y += (y[i] - y_mu)**2
        
        corr = top_term/np.sqrt(btm_term_x * btm_term_y)
        return corr

    def pdf(self,round_by=2):
        sorted_data = sorted(np.round(self.vector,round_by),reverse=False)
        n = len(sorted_data)
        unique_sorted_data = np.unique(sorted_data)
        counter = collections.Counter(sorted_data)
        self.vals, self.cnt = zip(*counter.items())
        self.probVector = [x/n for x in self.cnt]
        plt.scatter(self.vals,self.probVector)
        plt.show()
    
    def pdf_linearBinning(self):
        if not self.probVector:
            self.pdf(self.vector)
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, self.probVector,'o')
        plt.show()
    
    def pdf_log_binning(self,col):
        if not self.probVector:
            self.pdf(self.vector)
        inMax, inMin = max(self.probVector), min(self.probVector)
        logBins = np.logspace(np.log10(inMin),np.log10(inMax))
        # degree, count
        self.vals, log_bin_edges = np.histogram(self.probVector,bins=logBins,
                                       density=True, range=(inMin, inMax))
        plt.title(f"Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, log_bin_edges[:-1], 'o')
        plt.show()

    def ctl(self, samples=1000):
        res = []
        n = len(self.vector)-1

        for dataPoint in range(samples):
            idxVector = [ 
                        vector[np.random.randint(0,n)], 
                        vector[np.random.randint(0,n)],
                        vector[np.random.randint(0,n)]
                        ]
            rs = np.sum(idxVector) // len(idxVector)
            res.append(rs)
        plt.hist(res)
        plt.show()
        self.ctl_values = np.hist(rs)

    def get_entropy(self):
    
        h = collections.defaultdict(int)
        
        for node in self.vector:
            h[node] += 1
        p = []
        for x in h.keys():
           p.append((h[x] / self.n))

        self.p = p
        self.entropy = round(-np.sum(p * np.log2(p)),2)
        print(self.entropy)
    
    def get_variance(self):
        self.variance = np.sum([(x - self.vector_mu)**2 for x in self.vector]) / self.n
    
    def norm_vector(self):
        v_min = np.min(self.vector)
        v_max = np.max(self.vector)
        v_max_min = v_max - v_min
        self.vector_norm = [(x-v_min)/(v_max_min) for x in self.vector]
    
    def strings_to_time(self, c):
        def helper(c):
            nums = ['0','1','2','3','4','5','6','7','8','9']
            flag = 0
            ans = ''
            for i in c:
                if i in nums:
                    flag = True
                    ans += i
                elif flag:
                    break
            if len(ans) > 0:
                return int(ans)
            return 0

        return c.apply(lambda x: helper(x))

    def vector_to_ints(self, vector):
        unique = np.unique(vector)
        look_up = collections.defaultdict()
        for idx, val in enumerate(unique):
            look_up[val] = idx
        return [look_up[x] for x in vector]

class point(node):
    def __init__(self,x,y,label) -> None:
        self.n1 = node.__init__(self, x)
        self.n2 = node.__init__(self, y)
        self.x = x 
        self.y = y 
        self.label = label

        
        

    