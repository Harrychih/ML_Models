""" 
HAC model implementations 
"""

import numpy as np
from scipy.spatial import distance_matrix

class Model(object):
    """ Abstract model object."""

    def __init__(self):
        raise NotImplementedError()

    def fit_predict(self, X):
        """ Predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()



class AgglomerativeClustering(Model):

    def __init__(self, n_clusters = 2, linkage = 'single'):
        """
        Args:
            n_clusters: number of clusters
            linkage: linkage criterion
        """
        # Initializations etc. go here.

        self.n_clusters = n_clusters
        self.linkage = linkage

    def return_result(self, n, input):
        result = [-1] * n
        group_number = 0
        for i in input:
            for j in input[i]:
                result[j] = group_number
            group_number += 1
        return result
            
    def fit_predict(self, X):
        """ Fit and predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!

        n = X.shape[0]
        cluster = dict()
        l = []
        for i in range(n):
            a = [i]
            cluster[i] = a
            l.append(i)
        l = np.array(l)

        d = distance_matrix(X, X)
        d[d==0] = np.inf  
        num = n

        while num != self.n_clusters:
            # Break ties rules
            indexes = np.argwhere(d==np.min(d))
            min_index = indexes[:,0].argsort()[0]
            i = indexes[min_index][0]
            j = indexes[min_index][1]

            index = sorted(list(set(cluster[i] + cluster[j])))
            cluster[i] = index
            cluster[j] = index
            for x in cluster[i]:
                # set value to inf so that it does not interfere minimum searching
                cluster[x] = index
                d[x][i] = np.inf
                d[i][x] = np.inf
                d[x][j] = np.inf
                d[j][x] = np.inf
            temp = []
            unique = dict()
            for key, val in cluster.items():
                if val not in temp:
                    temp.append(val)
                    unique[key] = val        
            num = len(unique)
            temp = np.delete(l, index)
            if num == self.n_clusters:
                result = self.return_result(n, unique)
                return result
            for x in temp:
                tmp = np.delete(d[x], temp)
                # three different types of linkages
                if self.linkage == 'single':
                    val = np.min(tmp)
                if self.linkage == 'complete':
                    val = np.max(tmp)
                if self.linkage == 'average':
                    val = np.sum(tmp, dtype=np.float64) / (tmp.size * x.size)
                for y in cluster[i]:
                    d[x][y] = val
                    d[y][x] = val
