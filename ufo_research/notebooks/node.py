import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class vect():
    def __init__(self,vector,toInt=False,timeChange=False):
        self.vector = np.array(vector)
        self.n = len(self.vector)
        # If qualitative vector
        if toInt:
            self.vector_to_ints()
        if timeChange:
            self.strings_to_time()

        self.vector_mu = np.mean(self.vector)
        self.distro = np.histogram(self.vector)        
        self.pdf()
        
        

    def basic_stats(self):
        self.pdf_log_binning()
        self.get_entropy()
        self.get_cdf()
        self.get_variance()
        self.std = np.sqrt(self.variance)
        self.norm_vector()
        print('entropy:', self.entropy)
        print('variance:', self.variance)
        print('vector_mu:', self.vector_mu)
        print('std:', self.std)   

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

    def pdf(self):
        sorted_data = sorted(self.vector,reverse=False)
        n = len(sorted_data)
        unique_sorted_data = np.unique(sorted_data)
        counter = collections.Counter(sorted_data)
        self.vals, self.cnt = zip(*counter.items())
        self.probVector = [x/n for x in self.cnt]
        plt.scatter(self.vals,self.probVector)
        plt.title(f"PDF: Linear Binning & Scaling")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.show()
    
    def pdf_linearBinning(self):
        if not self.probVector:
            self.pdf(self.vector)
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, self.probVector,'o')
        plt.show()
    
    def pdf_log_binning(self):
        if not self.probVector:
            self.pdf(self.vector)

        inMax, inMin = max(self.probVector), min(self.probVector)
        self.logBins = np.logspace(np.log10(inMin),np.log10(inMax))
        # degree, count
        self.hist_cnt, self.log_bin_edges = np.histogram(self.probVector,bins=self.logBins,
                                       density=True, range=(inMin, inMax))
        plt.title(f"Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        n = np.sum(self.hist_cnt)
        self.log_prob_vector = [x/n for x in self.hist_cnt]
        plt.plot(self.hist_cnt, self.log_prob_vector[::-1], 'o')
        plt.show()

    def get_cdf(self):
        values = np.array(self.probVector)
        cdf = values.cumsum() / values.sum()
       # cdf = np.cumsum(probVector)
        self.ccdf = 1-cdf
        self.cdf = cdf 
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"Cumulative Distribution")
        plt.ylabel("P(K) >= K")
        plt.xlabel("K")
        plt.plot(cdf[::-1])
        plt.show()

    def get_ctl(self, samples=1000):
        res = []
        n = len(self.vector)-1

        for dataPoint in range(samples):
            idxVector = [ 
                        self.vector[np.random.randint(0,n)], 
                        self.vector[np.random.randint(0,n)],
                        self.vector[np.random.randint(0,n)]
                        ]
            rs = np.sum(idxVector) // len(idxVector)
            res.append(rs)
        plt.hist(res)
        plt.show()
        self.ctl_values = np.histogram(rs)

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
    
    def strings_to_time(self):
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

        self.vector = [helper(x) for x in self.vector] 

    def vector_to_ints(self):
        unique = np.unique(self.vector)
        look_up = collections.defaultdict()
        for idx, val in enumerate(unique):
            look_up[val] = idx
        self.vector = [look_up[x] for x in self.vector]

class point(vect):
    def __init__(self,x,y,label) -> None:
        self.n1 = vect.__init__(self, x)
        self.n2 = vect.__init__(self, y)
        self.x = x 
        self.y = y 
        self.label = label
        self.pearsonR = self.n1.get_corr(y)

        
        

    