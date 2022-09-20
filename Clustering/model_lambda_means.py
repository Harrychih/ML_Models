""" 
lambda means model implementations
"""

import numpy as np
from scipy.spatial import distance


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures


    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class LambdaMeans(Model):

    def __init__(self, *, nfeatures, lambda0):
        super().__init__(nfeatures)
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
        """
        # TODO: Initializations etc. go here.
        self.n = nfeatures
        self.l = lambda0
        

    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        # Implement
        N = X.shape[0]
        X = X.A
        prototype_vector = np.zeros((N, self.n))
        prototype_vector[1] = np.mean(X, axis=0)

        if self.l != 0.0:
            lmda = self.l
        else:
            dis = [0] * N
            for i in range(N):
                cal = distance.euclidean(X[i], prototype_vector[1])
                dis[i] = cal
            lmda = sum(dis) / len(dis)

        # cluster indicator
        r = [1] * N
        K = 1
        
        l = []
        for i in range(N):
            l.append(i) 

        # start of iteration
        for x in range(iterations):
            for i in range(N):
                min = np.inf
                index = 0
                index_k = 0
                # find the cluster that has the closest euclidean distance and keep track of its index
                for k in range(1,K+1):
                    dist = distance.euclidean(X[i], prototype_vector[k])
                    if dist < min:
                        min = dist
                        index = i
                        index_k = k
                    if dist == min and k < index_k: # tie breaking
                        index_k = k
                        index = i
            # E step
                if min > lmda:
                    K = K + 1
                    prototype_vector[K] = X[index]
                    r[index] = K
                else: # (1)
                    r[index] = index_k
            # M step
            cluster = dict()
            for k in range(1,K+1):
                cluster[k] = []
            # form clusters
            for i in range(N):
                cluster[r[i]].append(i)
            
            # recalculating prototype vector to see centroid positions
            for k in range(1,K+1):
                tmp = np.delete(l, cluster[k])
                lis = np.delete(X, tmp, axis=0)
                prototype_vector[k] = np.mean(lis, axis=0) # (2) 
            # both (1) and (2) together work to handle empty clusters

        # store the cluster centroids
        self.data = []
        for k in range(1,K+1):
            self.data.append(prototype_vector[k])
        self.data = np.array(self.data)
        self.k = K

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!
        # padding to make dimensions match
        pad = max(0, self.n-X.shape[1])
        X = np.pad(X.A, ((0,0),(0,pad)), 'constant')

        # prediction
        N = X.shape[0]
        result = []
        for i in range(N):
            label = 0
            min = np.inf
            for k in range(self.k):
                dist = distance.euclidean(X[i], self.data[k])
                if dist < min:
                    min = dist
                    label = k
            result.append(label)
        return result