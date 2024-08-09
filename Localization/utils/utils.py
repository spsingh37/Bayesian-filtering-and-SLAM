import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

from system.RobotState import *

def wrap2Pi(input):
    phases =  (( -input + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

    return phases

def func(x):
    
    G1 = np.array([[0,0,1],
                    [0,0,0],
                    [0,0,0]])
    
    G2 = np.array([[0,0,0],
                    [0,0,1],
                    [0,0,0]])
    
    G3 = np.array([[0,-1,0],
                    [1,0,0],
                    [0,0,0]])

    X = expm(x[0]*G1+x[1]*G2+x[2]*G3)
    y = np.zeros(3)

    y[0] = X[0,2]
    y[1] = X[1,2]
    y[2] = np.arctan2(X[1,0],X[0,0])

    return y

def unscented_propagate(mean, cov, kappa):
    n = np.size(mean)

    x_in = np.copy(mean)
    P_in = np.copy(cov)

    # sigma points
    L = np.sqrt(n + kappa) * np.linalg.cholesky(P_in)
    Y_temp = np.tile(x_in, (n, 1)).T
    X = np.copy(x_in.reshape(-1, 1))
    X = np.hstack((X, Y_temp + L))
    X = np.hstack((X, Y_temp - L))

    w = np.zeros((2 * n + 1, 1))
    w[0] = kappa / (n + kappa)
    w[1:] = 1 / (2 * (n + kappa))

    new_mean = np.zeros((n, 1))
    new_cov = np.zeros((n, n))
    Y = np.zeros((n, 2 * n + 1))
    for j in range(2 * n + 1):
        Y[:,j] = func(X[:,j])
        new_mean[:,0] = new_mean[:,0] + w[j] * Y[:,j]

    diff = Y - new_mean
    for j in range(np.shape(diff)[1]):
        diff[2,j] = wrap2Pi(diff[2,j])

    w_mat = np.zeros((np.shape(w)[0],np.shape(w)[0]))
    np.fill_diagonal(w_mat,w)
    new_cov = diff @ w_mat @ diff.T
    cov_xy = (X - x_in.reshape(-1,1)) @ w_mat @ diff.T

    return new_cov


def lieToCartesian(mean, cov):

    mean_matrix = np.eye(3)
    R = np.array([[np.cos(mean[2]),-np.sin(mean[2])],[np.sin(mean[2]),np.cos(mean[2])]])
    mean_matrix[0:2,0:2] = R
    mean_matrix[0,2] = mean[0]
    mean_matrix[1,2] = mean[1]

    kappa = 2
    X = logm(mean_matrix)
    x = np.zeros(3)
    x[0] = X[0,2]
    x[1] = X[1,2]
    x[2] = X[1,0]

    mu_cart = np.array([mean[0], mean[1], mean[2]])
    Sigma_cart = unscented_propagate(x, cov, kappa)

    return mu_cart, Sigma_cart

def mahalanobis(state, ground_truth, filter_name, Lie2Cart):
    # Output format (7D vector) : 1. Mahalanobis distance
    #                             2-4. difference between groundtruth and filter estimation in x,y,theta 
    #                             5-7. 3*sigma (square root of variance) value of x,y, theta 
    results = np.zeros((7,1))
    if filter_name != "InEKF":
        mu = state.getState()
        Sigma = state.getCovariance()
        ground_truth[2]=wrap2Pi(ground_truth[2])
        diff = ground_truth - mu
        diff[2] = wrap2Pi(diff[2])
        results[0,0] = (diff).T @ (np.linalg.inv(Sigma) @ diff)
        results[1:4,0] = diff
        results[4] = 3*np.sqrt(Sigma[0,0])
        results[5] = 3*np.sqrt(Sigma[1,1])
        results[6] = 3*np.sqrt(Sigma[2,2])
        

        return results.reshape(-1)
    else:
        mu = state.getState()
        if Lie2Cart:
            Sigma_cart = state.getCartesianCovariance()
            mu_cart = state.getCartesianState()
        else:
            mu_cart = np.zeros((np.shape(mu)))
            Sigma_cart = np.eye(np.shape(mu)[0])
        ground_truth[2]=wrap2Pi(ground_truth[2])
        diff = ground_truth - mu_cart
        diff[2] = wrap2Pi(diff[2])
        results[0,0] = (diff).T @ (np.linalg.inv(Sigma_cart) @ diff)
        results[1:4,0] = diff
        results[4] = 3*np.sqrt(Sigma_cart[0,0])
        results[5] = 3*np.sqrt(Sigma_cart[1,1])
        results[6] = 3*np.sqrt(Sigma_cart[2,2])
        

        return results.reshape(-1)
    

def plot_error(results, gt):
    num_data_range = range(np.shape(results)[0])

    gt_x = gt[:,0]
    gt_y = gt[:,1]

    plot2 = plt.figure(2)
    plt.plot(num_data_range,results[:,0])
    plt.plot(num_data_range, 7.81*np.ones(np.shape(results)[0]))
    plt.title("Chi-square Statistics")
    plt.legend(["Chi-square Statistics", "p = 0.05 in 3 DOF"])
    plt.xlabel("Iterations")

    plot3,  (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title("Deviation from Ground Truth with 3rd Sigma Contour")
    ax1.plot(num_data_range, results[:,1])
    ax1.set_ylabel("X")
    ax1.plot(num_data_range,results[:,4],'r')
    ax1.plot(num_data_range,-1*results[:,4],'r')
    
    ax1.legend(["Deviation from Ground Truth","3rd Sigma Contour"])
    ax2.plot(num_data_range,results[:,2])
    ax2.plot(num_data_range,results[:,5],'r')
    ax2.plot(num_data_range,-1*results[:,5],'r')
    ax2.set_ylabel("Y")

    ax3.plot(num_data_range,results[:,3])
    ax3.plot(num_data_range,results[:,6],'r')
    ax3.plot(num_data_range,-1*results[:,6],'r')
    ax3.set_ylabel("theta")
    ax3.set_xlabel("Iterations")
    
    plt.show()


def main():

    i = -7
    j = wrap2Pi(i)
    print(j)
    

if __name__ == '__main__':
    main()