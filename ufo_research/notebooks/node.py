import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class vect():
    
    def __init__(self,name):
        self.name = name 
        self.pdf_from_mu_vect = None 

    def set_up(self,vector,toInt=False,timeChange=False):
        self.vector = np.array(vector)
        self.n = len(self.vector)
        # If qualitative vector
        if toInt:
            self.vector_to_ints()
        if timeChange:
            self.strings_to_time()

        c = collections.Counter(self.vector)
        self.vals, self.cnt = list(zip(*c.items()))
        self.vector_mu = np.mean(self.cnt)
        self.distro = c   
            
    def basic_stats(self,show=True):
        self.get_variance()
        self.std = np.sqrt(self.variance)
        self.norm_vector()
        self.pdf(show)
        self.pdf_from_mu(show)
        self.pdf_log_binning(show)
        self.get_entropy()
        self.get_cdf(show)
        
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
        for i in range(self.n):
            top_term += (x[i] - x_mu) * (y[i] - y_mu)
            btm_term_x += (x[i] - x_mu)**2
            btm_term_y += (y[i] - y_mu)**2
        
        corr = top_term/np.sqrt(btm_term_x * btm_term_y)
        return corr

    def pdf(self,show=False,title=False):
        #sorted_data = sorted(self.vector,reverse=False)
        if not self.cnt or not self.vals:
            counter = collections.Counter(self.vector)
            self.vals, self.cnt = zip(*counter.items())
        
        n = np.sum(self.cnt)
        self.probVector = [x/n for x in self.cnt]

        if show:
            plt.scatter(self.vals,self.probVector)
            if not title:
                plt.title(f"PDF: Linear Binning & Scaling")
            else:
                plt.title(title)
                
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
    
    def pdf_log_binning(self,show=True):
        if not self.probVector:
            self.pdf(self.vector)

        inMax, inMin = max(self.probVector), min(self.probVector)
        self.logBins = np.logspace(np.log10(inMin),np.log10(inMax))
        # degree, count
        self.hist_cnt, self.log_bin_edges = np.histogram(self.probVector,bins=self.logBins,
                                    density=True, range=(inMin, inMax))
       
        n = np.sum(self.hist_cnt)

        self.log_prob_vector = [x/n for x in self.hist_cnt]

        if show:
            plt.title(f"Log Binning & Scaling")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('K')
            plt.ylabel('P(K)')
            plt.plot(self.hist_cnt, self.log_prob_vector[::-1], 'o')
            plt.show()

    def get_cdf(self,show=True):
        values = np.array(self.probVector)
        cdf = values.cumsum() / values.sum()
       # cdf = np.cumsum(probVector)
        self.ccdf = 1-cdf
        self.cdf = cdf 
        if show:
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
        return self.variance
    
    def norm_vector(self,x=False):
        if not x:
            x = self.vector
        v_min = np.min(x)
        v_max = np.max(x)
        v_max_min = v_max - v_min
        self.vector_norm = [(x-v_min)/(v_max_min) for x in x]
        return self.vector_norm
    
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
    
    def dist(self,p1,p2):
        res = (p2.x - p1.x)**2 + (p2.y - p1.y)**2
        return round(np.sqrt(res),4)

    def pdf_from_mu(self, show=True, idx=-1):
        '''        
        f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))

        Where:

        x is the value of the random variable
        μ is the mean of the distribution
        σ is the standard deviation of the distribution
        e is the mathematical constant e ≈ 2.71828
        
        '''
        if not self.pdf_from_mu_vect:
            c = collections.Counter(self.vector)
            vals, cnt = zip(*c.items())
            cnt = self.norm_vector(cnt)
            n = np.sum(cnt)
            mu = n/len(cnt)
            std = np.std(cnt)
            self.pdf_from_mu_vect = [x/n for x in cnt]
        if show and idx >-1:
            print("PDF_VALUE:", self.pdf_from_mu_vect[idx])
            
    def create_corr_vectors(self,n,corr):
        # Generate the first random vector from a normal distribution
        x = np.random.normal(loc=0, scale=1, size=n)
            # Generate the second random vector from a normal distribution
        y = np.random.normal(loc=0, scale=1, size=n)
            # Create a third vector with the desired correlation
        z = (y + corr) * np.std(x) * (x - np.mean(x))
        np.corrcoef(x,z)
        return x,z
    
    def get_coverance(self,v2):
        n = len(self.vector)
        y_mu = np.mean(v2)
        coverance = np.sum([(x - self.vector_mu)*(y - y_mu) 
                            for x,y in zip(self.vector,v2)]) / n-1
        return coverance
    
    def get_slope(self,v2):
        slope = self.get_coverance(v2) / self.get_variance()
        self.slope = slope
        return slope

class point(vect):
    def __init__(self,x,y,label) -> None:
        super().__init__(label)

        self.n1 = vect(label)
        self.n2 = vect(label)
        self.n1.set_up(x)
        self.n2.set_up(y)
        
        self.x = x 
        self.y = y
        self.n = len(x)
        self.m = len(y) 
        

        self.nodeList = None 
        self.adjList = None
        self.scatterGraph = None 
        self.scatterGraphNorm = None
        self.graph = None
        self.distanceVector = None
        
        # Creating instances of the parent class
        self.label = label
        self.pearsonR = self.n1.get_corr(y)
        self.coverance = self.n1.get_coverance(self.n2.vector)
        self.slope = self.n1.get_slope(self.n2.vector)

    # AI Algorithms
    def linear_regression(self):
         # y_hat = w.X + b
        n = len(self.x)
        x_mu = self.n1.vector_mu
        y_mu = self.n2.vector_mu
        

        top_term = 0
        btm_term = 0

        for i in range(n):
            top_term += (self.x[i] - x_mu) * (self.y[i] - y_mu)
            btm_term += (self.x[i] - x_mu)**2

        m = top_term/btm_term
        b = y_mu - (m * x_mu)

        
        print (f'm = {m} \nb = {b}')


        max_x = np.max(self.x) + 10
        min_x = np.min(self.y) - 10
        x_delta = np.linspace (min_x, max_x, 10)

        y_delta = b + m * x_delta

        plt.scatter(self.x,self.y)
        plt.plot(x_delta,y_delta,'ro')
        plt.show()
        return y_delta              

    def knn_predict(self,target,knnSize=5):
        '''
            vector 1D Node array
        '''
        # Define a list of 30 colors
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'black',
          'olive', 'navy', 'teal', 'magenta', 'maroon', 'coral', 'gold', 'lime', 'indigo',
          'peru', 'slateblue', 'sienna', 'rosybrown', 'mediumvioletred', 'cadetblue', 'crimson',
          'darkcyan', 'deeppink', 'firebrick', 'forestgreen', 'fuchsia','blue',
          'peru', 'slateblue']

        # Define a list of 30 marker symbols
        shapes = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd', 'p', 'P', '*', 'h', 'H', 'X', '|', '_']

    
        marker_dict = {i: shapes[i] for i in range(len(shapes)-1)}
        colors_dict = {i: colors[i] for i in range(len(colors)-1)}


        # preprocess
        self.knn_init()
        self.distanceVector = collections.defaultdict()
        cord_vector = list(self.graph.values())
        # update graph
        p1 = target
        dx,dy,dl = [],[],[]
        for p2 in self.nodeList:
            self.distanceVector[(p2.x,p2.y)] = self.dist(p1,p2)
            dx.append(p2.x)
            dy.append(p2.y)
            dl.append(p1.x)
        # TODO: optimize lookup with headpq
        delta_values = list(self.distanceVector.values())
        delta_keys = list(self.distanceVector.keys())
        delta = sorted(list(zip(delta_values,delta_keys,dl)),
                        key= lambda x:x[0], reverse=False)
        
        # TODO: CLUSTER GROUPs
        fig, ax = plt.subplots()
        for i in range(len(dx)):
            ax.scatter(dx[i], dy[i])#, marker=marker_dict[dx[i]],color=colors_dict[dy[i]])
        ax.scatter(p2.x,p2.y,c='red',marker="o",  s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()
        self.knn_predict_res = delta[0:knnSize]
        return delta[0:knnSize]

    def knn_init(self):
        self.create_node_list()
        self.create_graph()
        
    def create_node_list(self):
        x,y = self.x, self.y
        nodeList = [] 
        adjList = collections.defaultdict()
        for idx in range(self.n):
            dx = self.x[idx]
            dy = self.y[idx]
            delta = node(   dx, 
                            dy,
                            idx,
                            self.label         
                        )
            nodeList.append(delta)
            adjList[(dx,dy)] = delta
        
        self.nodeList = nodeList
        self.adjList = adjList
    
    def create_graph(self):
        if not self.nodeList:
            return ('No node list. Complie first')
        
        graph = collections.defaultdict(list)

        for node in self.nodeList:
            x,y = node.x, node.y 
            graph[(x,y)].append(node)
        
        delta = list(graph.keys())
        x = [i[0] for i in delta]
        y = [i[1] for i in delta]
        norm_x, norm_y = self.norm_vector_2D(x,y)
        self.scatterGraph = (x,y)
        self.scatterGraphNorm = (norm_x, norm_y)
        self.graph = graph
        return graph
    
    # Helpers 
    def norm_vector_2D(self, x,y):
        x = self.norm_vector(x)
        y = self.norm_vector(y)
        return x,y

    def norm_vector(self, vector):
        v_min = np.min(vector)
        v_max = np.max(vector)
        v_max_min = v_max - v_min
        return [(x-v_min)/(v_max_min) for x in vector]

    def dist(self,p1,p2):
        res = (p2.x - p1.x)**2 + (p2.y - p1.y)**2
        return round(np.sqrt(res),4) 

    def remove_extreme_outlier(self,r=False):
        '''
        TODO: Generalize a concept of 'extreme'
        r set True processes y
        '''
        x_y = list(zip(self.x, self.y))

        if not r:
            x_max = max(x_y, key=lambda x: x[0])
            x_max_idx = x_y.index(x_max)
            x_y.pop(x_max_idx)

        else:
            y_max = max(x_y, key=lambda x: x[1])
            y_max_idx = x_y.index(y_max)
            x_y.pop(y_max_idx)

        dx, dy = list(zip(*x_y))
        self.x, self.y = dx, dy
        self.n1.set_up(dx)
        self.n2.set_up(dy)
        
    
class node():
    
    def __init__(self,x,y,idx,label) -> None:
        self.x = x 
        self.y = y 
        self.idx = idx
        self.label = label

class LogisticR:
    def __init__(self,X,y) -> None:
        self.X = X
        self.y = y
        self.iterations = 1000
        self.learning_rate = 0.01

    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    # AI
    def logistic_regression_fit(self):

        X = np.array(self.X)
        y = np.array(self.y)
        m = len(y)
        # Add col of 1 for bias term
        X = np.insert(X, 0, 1, axis=1)
        # init weights to 0
        self.w = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)

            cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            dw = (1/ m) * np.dot(X.T, (y_pred - y))

            self.w -= self.learning_rate * dw

            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    
    def log_predict(self, x):
        x = np.array(x)
        x = np.insert(x,0,1,axis=1)
        
        z = np.dot(x, self.w) 
        y_pred = self.sigmoid(z)

        self.y_pred = y_pred
        return np.round(y_pred)