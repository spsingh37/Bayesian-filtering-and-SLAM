#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import kalman_filter


class myStruc:
    pass


#
# Example for tracking a single target using a Kalman filter
#
if __name__ == "__main__":
    # First simulate a target that moves on a curved path; we assume ownship is at the origin (0, 0)
    # and receives direct but noisy (x, y) measurements of the target location. There is no knowledge
    # of the target motion, hence, we assume a random walk motion model

    # ground truth data
    gt = myStruc()
    gt.x = np.arange(-5, 5.1, 0.1)
    gt.y = np.array(1 * np.sin(gt.x) + 3)

    # measurements
    R = np.diag(np.power([0.05, 0.05], 2))
    # Cholesky factor of covariance for sampling
    L = np.linalg.cholesky(R)
    z = np.zeros([2, len(gt.x)])

    for i in range(len(gt.x)):
        # sample from a zero mean Gaussian with covariance V
        noise = np.dot(L, np.random.randn(2, 1)).reshape(-1)
        z[:, i] = np.array([gt.x[i], gt.y[i]]) + noise

    # build the system
    sys = myStruc()
    sys.A = np.eye(2)
    sys.B = []
    sys.H = np.eye(2)
    sys.Q = 1e-3 * np.eye(2)
    sys.R = 0.05 ** 2 * np.eye(2)

    # initialize the state using the first measurement
    init = myStruc()
    init.x = np.zeros([2, 1])
    init.x[0, 0] = z[0, 0]
    init.x[1, 0] = z[1, 0]
    init.Sigma = 2 * np.eye(2)

    kf = kalman_filter(sys, init)
    x = []
    x.append(init.x)  # state
    for i in range(1, np.shape(z)[1], 1):
        kf.prediction()
        kf.correction(z[:, i].reshape([2, 1]))
        x.append(kf.x)

    x = np.array(x)

    # printing the RMSE
    gt_x = gt.x
    gt_y = gt.y
    pred_x = x[:, 0, :].flatten()
    pred_y = x[:, 1, :].flatten()

    # Calculate RMSE
    rmse_x = np.sqrt(np.mean((gt_x - pred_x) ** 2))
    rmse_y = np.sqrt(np.mean((gt_y - pred_y) ** 2))

    # Print RMSE
    print(f"RMSE for x: {rmse_x}")
    print(f"RMSE for y: {rmse_y}")
    
    
    # plotting
    fig = plt.figure()
    line1, = plt.plot(0, 0, '^', color='b', markersize=20)
    line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    line3, = plt.plot(x[:, 0, :], x[:, 1, :], '-r', linewidth=1.5)
    plt.legend([line1, line2, line3], [r'ownship', r'ground truth', r'Kalman filter'], loc='best')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.show()
