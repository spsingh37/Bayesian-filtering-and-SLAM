import numpy as np
from scipy.linalg import logm, expm
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import matplotlib.animation as ani
from riekf import Right_IEKF
from scipy.linalg import  expm

import cv2


def motion_model(R_t, w_tilde,dt):
    """
    :param R_t: State variable that shows sensor frame orientatiion wrt the world frame (3x3)
    :param w_tilde: gyroscope reading needs to be in lie algebra matrix form (3x3)
    :return:
    """
    R = np.dot(R_t, expm(w_tilde*dt) )
    return R


def measurement_error_matrix(g):
    # measurement error matrix
    H = np.array([[0, -g[2], g[1]],
                  [g[2], 0, -g[0]],
                  [-g[1], g[0], 0]], dtype=float)
    return H


def convert_row_pitch_yaw(R):

    if R[0,0]==0 and R[1,0]==0:
        beta = np.pi/2
        alpha = 0
        gamma = np.arctan(R[0,1], R[1,1])
    else:
        beta = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
        alpha = np.arctan2(R[1,0], R[0,0])
        gamma = np.arctan2(R[2,1], R[2,2] )

    return [alpha, beta, gamma]

if __name__ == "__main__":

    # Load data
    accel = np.loadtxt(open('data/a.csv'), delimiter="," )
    omega = np.loadtxt(open('data/omega.csv'), delimiter=",")
    dt = np.loadtxt(open('data/dt.csv'), delimiter=",").reshape(-1,1)
    gravity = np.loadtxt(open('data/gravity.csv'),delimiter=",").reshape(-1,1)
    euler_gt = np.loadtxt(open('data/euler_gt.csv'), delimiter=",")

    # build a system
    sys = {}
    # motion model noise covariance
    sys['Q'] = np.diag(np.power([0.01, 0.01, 0.01], 2))
    sys['A'] = np.eye(3)
    # A is idenity comes from the fact the state transton is idenity
    sys['f'] = motion_model
    sys['H'] = measurement_error_matrix
    sys['N'] = np.diag(np.power([0.01, 0.01, 0.01], 2))


    iekf_filter = Right_IEKF(sys)  # create an RI-EKF object
    R = []

    skip = 50
    for i in range(len(accel)):
        # predict next pose using given twist

        iekf_filter.prediction(omega[i,:],dt[i])
        # Note Y = accel[i, :]
        # g/b = gravity
        iekf_filter.correction(accel[i, :] , gravity)
        R.append(iekf_filter.X)


    euler_found = []
    for R_local in R:
        euler_found.append(convert_row_pitch_yaw(R_local))

    euler_found = np.array(euler_found)

    # plotting
    fig = plt.figure()
    time = np.cumsum(dt).reshape(-1,1)
    # for some reason euler_gt has 1278 elements so I'm not reading the last one
    # to make dims match and to be able to plot

    plt.plot(time,euler_found[:, 0], label='Predicted')
    plt.plot(time, euler_gt[:-1, 0], label='GT')
    plt.xlabel(r'Time')
    plt.ylabel(r'Position')
    plt.legend()
    plt.title('Roll')
    plt.show()

    fig1 = plt.figure()
    plt.plot(time,euler_found[:, 1], label='Predicted')
    plt.plot(time, euler_gt[:-1, 1], label='GT')
    plt.xlabel(r'Time')
    plt.ylabel(r'Position')
    plt.legend()
    plt.title('Pitch')
    plt.show()

    fig2 = plt.figure()
    plt.plot(time, euler_found[:, 2], label='Predicted')
    plt.plot(time, euler_gt[:-1, 2], label='GT')
    plt.xlabel(r'Time')
    plt.ylabel(r'Position')
    plt.legend()
    plt.title('Yaw')
    plt.show()


