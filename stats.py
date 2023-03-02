import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt

class formulas:
    # Dataframe management
    def __init__(self,df) -> None:
        self.df = df.dropna()
        self.n = len(self.df)
        self.nodeList = [] 
        self.entropy_res = collections.defaultdict(int)
        self.prob_res = collections.defaultdict(int)
        self.cols = list(self.df.columns.values)
        self.graph = collections.defaultdict(list)
        self.probVector = False
    
    def setCol(self,col):

        self.col = self.df[col]

    def createX(self, colName):
        self.x = self.df[colName]
        self.x_name = colName
        self.x_mu = np.mean(self.x)
    
    def createY(self, colName):
        self.y = self.df[colName]
        self.y_name = colName
        self.y_mu = np.mean(self.y)

    def set_x_y(self, x, y):
        self.createX(x)
        self.createY(y)
    
    def corr(self,x,y):
        self.set_x_y(x,y)
        top_term = 0
        btm_term_x = 0
        btm_term_y = 0
        

        n = len(self.x)
        m = len(self.y)

        if n!=m:
            print('woof, cols not equal length')
            return 
        
        for i in range(n):
            top_term += (self.x.iloc[i] - self.x_mu) * (self.y.iloc[i] - self.y_mu)
            btm_term_x += (self.x.iloc[i] - self.x_mu)**2
            btm_term_y += (self.y.iloc[i] - self.y_mu)**2
        
        self.corr = top_term/np.sqrt(btm_term_x * btm_term_y)
        print(self.corr)
        return self.corr

    def pdf(self,col):
        sorted_data = sorted(self.df[col],reverse=False)
        n = len(sorted_data)
        unique_sorted_data = np.unique(sorted_data)
        counter = collections.Counter(sorted_data)
        self.vals, self.cnt = zip(*counter.items())
        self.probVector = [x/n for x in self.cnt]
    
    def pdf_linearBinning(self,col):
        if not self.probVector:
            self.pdf(col)
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, self.probVector,'o')
        plt.show()
    
    def pdf_log_binning(self,col):
        if not self.probVector:
            self.pdf(col)
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

    def ctl(self, col, samples):
        res = []
        n = len(self.df[col])
        for dataPoint in range(samples):
            idxVector = [ np.random.randint(0,n),np.random.randint(0,n),np.random.randint(0,n)]

            randomSample = 0
            for idx in idxVector:
                randomSample +=  idx
            randomSample /= len(idxVector)
            res.append(randomSample)
        plt.hist(res)
        plt.show()

    def entropy(self, col):
    
        n = len(self.df[col])
        h = collections.defaultdict(int)
        
        for node in self.df[col]:
            h[node] += 1
        p = []
        for x in h.keys():
           p.append((h[x] / n))

        self.p = p
        entropy = round(-np.sum(p * np.log2(p)),2)
        print(entropy)
        self.entropy_res[col] = entropy
        self.prob_res[col] = p

    def dist(self,p1,p2):
        res = (p2.x - p1.x)**2 + (p2.y - p1.y)**2
        return round(np.sqrt(res),4)
        
    def init_knn(self,clusterSize,vals):
        '''
        vals must be x,y only
        TODO: create dynamic vals ds
        '''
        self.create_node_list(vals)
        self.create_graph()
        
    def create_node_list(self,vals,label='County'):
        x,y = vals[0], vals[1]
        names = [x,y]
        nodeList = [] 
        adjList = collections.defaultdict()
        for idx in range(self.n):
            delta = node(   self.df.iloc[idx][x], 
                            self.df.iloc[idx][y],
                            idx,
                            self.df,
                            names,
                            label         
                        )
            nodeList.append(delta)
            adjList[idx] = delta
        
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
        
    def norm_vector_2D(self, x,y):
        x = self.norm_vector(x)
        y = self.norm_vector(y)
        return x,y

    def norm_vector(self, vector):
        v_min = np.min(vector)
        v_max = np.max(vector)
        v_max_min = v_max - v_min
        delta = [(x-v_min)/(v_max_min) for x in vector]
        self.delta = delta
        return delta

    def insert_knn(self, node, normalized=False):
        if not normalized:
            node.label = 'Red'
            self.graph[(node.x,node.y)].append(node)
    
    def predict(self, vector, knnSize=5):
        '''
            vector 1D Node array
        '''
        self.distanceVector = collections.defaultdict()
        cord_vector = list(self.graph.values())
        # update graph
        while vector:
            p1 = vector.pop()
            for p2 in self.nodeList:
                self.distanceVector[(p2.x,p2.y)] = self.dist(p1,p2)
        # TODO: optimize lookup with headpq
        delta_values = list(self.distanceVector.values())
        delta_keys = list(self.distanceVector.keys())
        delta = sorted(list(zip(delta_values,delta_keys)),
                        key= lambda x:x[0], reverse=False)
        
        return delta[0:knnSize]

    def vector_to_ints(self,col):
        v = self.df[col]
        counts = collections.Counter(v)
        unique_list = list(counts.keys())
        range_of_values = len(unique_list)
        delta_vector = collections.defaultdict()
        i = 0

        for x in unique_list:
            delta_vector[x] = i
            i+=1

        for idx, val in enumerate(v):
            v[idx] = delta_vector[x] 
        
        return v


        

class node(formulas):
    
    def __init__(self,x,y,idx,df,names,label) -> None:
        self.x = x 
        self.y = y 
        self.idx = idx
        self.names = names
        self.label = label
        # invoking the __init__ of the parent class
        formulas.__init__(self, df)
        
        
        

    



# df = pd.read_csv('stats_py_ai_ml/data/nyc.csv')
# jedi = formulas(df)
# jedi.df['County'] = jedi.vector_to_ints('County')


# testNode = node(24,29,None,df,['TestNode'],'Brooklyn')
# jedi.init_knn(5,['White','Black'])
# jedi.insert_knn(testNode)
# jedi.predict([testNode])
# print(jedi.distanceVector)

#T1
# jedi.set_x_y('Poverty','ChildPoverty')
# jedi.pdf_linearBinning('Poverty')
# jedi.pdf_log_binning('Poverty')
# jedi.ctl('Poverty',1000)
# jedi.corr('Poverty','Income')
# print(df['Poverty'].corr(df['Income']))

#T2
# jedi.init_knn(5, ['Poverty', 'White'])
# print(jedi.adjList[0])
# node = jedi.adjList[0]
# print('x:',node.x,' y:',node.y, ' idx:', node.idx)
# print('names:', node.names)
# print('cols:', jedi.cols)
# node.entropy('Unemployment')