import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
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

class LogisticRegressionSGD(Model):

    def __init__(self, n_features, learning_rate = 0.1):
        super().__init__()
        # MLInitialize parameters, learning rate
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.coefficient = np.array([1] * n_features)
        # pass

    def fit(self, X, y):
        # Write code to fit the model
        X = X.toarray()
        # i = 0  # iteration times
        learning_rate = self.learning_rate
        for i in range(X.shape[0]):
            error = 0
            sigmoid_input = X[i, :].dot(self.coefficient)  # the value to input to sigmoid function

            y_pred_i = sigmoid(sigmoid_input)  # compute the sigmoid function output, dot product of sample x_i * w

            w_new = self.coefficient + learning_rate * np.multiply(X[i, :], float(y[i]-y_pred_i))  # formula w_j+1 = w_j + lambda * x_i * (y_i - y_predict)

            #y_pred_result = X.dot(self.coefficient)  # all predictions using w_old
            #for index in range(y_pred_result.shape[0]):  # no need to compute the error convergence, just epoch it
            #   error += y[index] - y_pred_result[index]  # calculate the total errors
            #print("error now of {} iteration is: {}".format(i, error))
            self.coefficient = w_new  # now w_new becomes w_old
            # i += 1
        return self.coefficient
        # pass

    def predict(self, X):
        # TODO: Write code to make predictions
        X = X.toarray()
        diffNumCol = max(0, self.n_features-X.shape[1])
        X = np.pad(X, ((0, 0), (0, diffNumCol)), 'constant')
        value = X.dot(self.coefficient)
        y_pred = sigmoid(value)
        # diffNumCol = max(0, self.n_features-X.shape[1])
        # X = np.pad(X, ((0, 0), (0, diffNumCol)), 'constant')
        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        y_predict = y_pred.reshape(len(y_pred), 1)
        return y_predict
       # pass

class LogisticRegressionNewton(Model):

    def __init__(self, n_features):
        super().__init__()
        # Initialize parameters
        self.n_features = n_features 
        self.coefficient = np.zeros([n_features, 1])

    def fit(self, X, y):
        X = X.toarray()
        p = np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            sigmoid_input_i = self.coefficient.T.dot(X[i,:].T)
            p[i,i] = sigmoid(sigmoid_input_i).dot((1-sigmoid(sigmoid_input_i)))
        Hessian = - X.T.dot(p).dot(X)
        sigmoid_input = self.coefficient.T.dot(X.T)
        # error = y - sigmoid(sigmoid_input)
        # print("error now of {} iteration is: {}".format(i, error))
        G_T = (y - sigmoid(sigmoid_input)).dot(X).T
        self.coefficient = np.subtract(self.coefficient,np.linalg.pinv(Hessian).dot(G_T)).reshape(self.n_features, 1)


    def predict(self, X):
        X = X.toarray()
        diffNumCol = max(0, self.n_features-X.shape[1])
        X = np.pad(X, ((0,0),(0,diffNumCol)), 'constant')
        value = X.dot(self.coefficient)
        y_pred = sigmoid(value)
        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        y_predict = y_pred.reshape(len(y_pred), 1)
        
        return y_predict
