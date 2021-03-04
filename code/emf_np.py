# Implementation of Explicit Factor Model, from paper "Explicit factor models for explainable recommendation based on phrase-level sentiment analysis"
# In Proceedings of the 37th international ACM SIGIR conference on Research and development in information retrieval - SIGIR 14, 83–92. https://doi.org/10.1145/2600428.2609579
# Zhang et al, 2014

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# Input:
# A: ratings, X: user att matrix, Y: item quality, m: users, n: hotels, r: latent features, r_: latent features,
# lambda_x, lambda_y, lambda_h, lambda_u, lambda_v: regularization coecients
# T: iterations

# Output:
# U1;U2; V;H1;H2
class EMF_NP(BaseEstimator, RegressorMixin):

    def __init__(self, p=8, r=0, total_r=100, lambda_x=0.05, lambda_y=0.05, lambda_h=0.05, lambda_u=0.05, lambda_v=0.05, T=2, Y=np.zeros(shape=(10,10))):
        """
        Perform EFM (Zhang et al 2014)
        Input:
        A: ratings, X: user att matrix, Y: item quality, p: number of aspects, r: latent features, r0: latent features,
        lambda_x, lambda_y, lambda_h, lambda_u, lambda_v: regularization coefficients
        T: iterations

        Output:
        U1;U2; V;H1;H2
        """
        self.p = p
        self.r = r
        self.r0 = int(total_r)-int(r)
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_h = lambda_h
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.total_r = total_r
        self.T = T
        self.Y = Y

    def fit(self, A, X, y=None):
        self.A = A
        self.m, self.n = A.shape
        self.X = X
        self.train()
        return self

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return self.mse()

    def train(self):
        # Initialize U1 (mxr), U2 (nxr), H1 (mxr0), H2 (nxr0), V (pxr) randomly
        # X (mxp), Y (nxp), A (mxn)

        self.U1 = np.random.random(size=(self.m, self.r))
        self.U2 = np.random.random(size=(self.n, self.r))
        self.H1 = np.random.random(size=(self.m, self.r0))
        self.H2 = np.random.random(size=(self.n, self.r0))
        self.V = np.random.random(size=(self.p, self.r))

        # Perform minimization algorithm for a number of iterations
        training_process = []
        for i in range(self.T):
            self.min_alg()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        X_pred = np.dot(self.U1, self.V.T)
        Y_pred = np.dot(self.U2, self.V.T)
        A_pred = np.dot(self.U1, self.U2.T) + np.dot(self.H1, self.H2.T)

        # pd.DataFrame(X_pred).to_csv('C:/Python/ArguAna/EFM/X_pred.csv', sep=',', index=False)
        # pd.DataFrame(Y_pred).to_csv('C:/Python/ArguAna/EFM/Y_pred.csv', sep=',', index=False)
        # pd.DataFrame(A_pred).to_csv('C:/Python/ArguAna/EFM/A_pred.csv', sep=',', index=False)

        #return training_process
        return mse

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.A.nonzero()
        predicted = self.full_matrix()
        #pd.DataFrame(predicted).to_csv('C:/Python/ArguAna/Data/Unnanotated/predictedZ.csv', sep=',', index=False)
        #pd.DataFrame(predicted).to_csv('/content/drive/My Drive/ABSA/data/predictedZ.csv', sep=',', index=False)
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.A[x, y] - predicted[x, y], 2)
        return np.sqrt(error/len(xs))

    def min_alg(self):
        #for m, n, rating in self.samples:
            # Create copy of original entries to update subsequent values
            V_ = self.V.copy()
            U1_ = self.U1.copy()
            U2_ = self.U2.copy()
            H1_ = self.H1.copy()

            # V_ = self.V
            # U1_ = self.U1
            # U2_ = self.U2
            # H1_ = self.H1

            # Apply imputation to deal with missing values

            # Z = self.A.copy()
            # xz, yz = np.where(self.A == 0)
            # #xz, yz = np.where(np.asnumpy(self.A) == 0) # here i use numpy, because zip is not working properly with cp
            #
            # for x, y in zip(xz, yz):
            #     Z[x,y] = np.dot(self.U1[x,:], self.U2[y,:].T) + np.dot(self.H1[x,:], self.H2[y,:].T)
            #     #Z[x, y] = np.dot(self.U1, self.U2.T)[x,y] + np.dot(self.H1, self.H2.T)[x,y]

            Z= self.U1.dot(self.U2.T) + self.H1.dot(self.H2.T)
            # restore original non zero values from A
            xz, yz = np.nonzero(self.A)
            for x, y in zip(xz, yz):
                Z[x,y] = self.A[x,y]


            # Update matrices

            V_num = self.lambda_x * np.dot(self.X.T,self.U1) + self.lambda_y * np.dot(self.Y.T, self.U2)
            V_den = np.dot(self.V, (self.lambda_x * np.dot(self.U1.T, self.U1) + self.lambda_y * np.dot(self.U2.T, self.U2) + self.lambda_v * np.identity(self.U1.shape[1])))
            self.V = V_ * np.sqrt(V_num / V_den)

            U1_num = np.dot(Z,self.U2) + self.lambda_x * np.dot(self.X, V_)
            U1_den = np.dot((np.dot(self.U1, self.U2.T) + np.dot(self.H1, self.H2.T)), self.U2) + np.dot(self.U1, (self.lambda_x * np.dot(V_.T, V_) + self.lambda_u * np.identity(V_.shape[1])))
            self.U1 = U1_ * np.sqrt(U1_num / U1_den)

            U2_num = np.dot(Z.T,U1_) + self.lambda_y * np.dot(self.Y, V_)
            U2_den = np.dot((np.dot(self.U2, U1_.T) + np.dot(self.H2, self.H1.T)), U1_) + np.dot(self.U2, (self.lambda_y * np.dot(V_.T, V_) + self.lambda_u * np.identity(V_.shape[1])))
            self.U2 = self.U2 * np.sqrt(U2_num / U2_den)

            H1_num = np.dot(Z, self.H2)
            H1_den = np.dot((np.dot(U1_, U2_.T) + np.dot(self.H1, self.H2.T)), self.H2) + self.lambda_h*self.H1
            self.H1 = self.H1 * np.sqrt(H1_num / H1_den)

            H2_num = np.dot(Z.T, H1_)
            H2_den = np.dot((np.dot(U2_, U1_.T) + np.dot(self.H2, H1_.T)), H1_) + self.lambda_h*self.H2
            self.H2 = self.H2 * np.sqrt(H2_num / H2_den)

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        # P = np.concatenate((self.U1, self.H1))
        # Q = np.concatenate((self.U2, self.H2))
        # #prediction = P[i, :].dot(Q[j, :].T)
        # prediction = P.dot(Q.T)

        alpha = 0.1 # scale between 0 and 1 that controls the trade off between feature based score and direct useritem ratings.
        k = 3 # most important features
        N = 5 # max rating in A

        X_pred = np.dot(self.U1, self.V.T)
        Y_pred = np.dot(self.U2, self.V.T)
        A_pred= np.dot(self.U1, self.U2.T) + np.dot(self.H1, self.H2.T)
        R = np.zeros(self.A.shape)

        # get column indices of highest values of X_pred
        highest_f = np.array([0,1,2])
        features_score=0
        for c in highest_f:
            features_score += X_pred[i,c] * Y_pred[j,c]
        R[i,j]= ( alpha * (features_score/(k*N)) ) + (1-alpha)*A_pred[i,j]
        return R

    def full_matrix(self):
        """
        Computer the full matrix using the resultant matrices U1,U2,H1,H2
        """
        alpha = 0.0001  # scale between 0 and 1 that controls the trade off between feature based score and direct useritem ratings.
        k = 3  # most important features
        N = 5  # max rating in A

        X_pred = np.dot(self.U1, self.V.T)
        Y_pred = np.dot(self.U2, self.V.T)
        A_pred = np.dot(self.U1, self.U2.T) + np.dot(self.H1, self.H2.T)

        return A_pred


