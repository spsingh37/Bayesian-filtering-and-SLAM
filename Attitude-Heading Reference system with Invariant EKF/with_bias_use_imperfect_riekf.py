
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
from imperfect_riekf import imperfect_Right_IEKF

# wraps to -pi to pi
def wrapToPI(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap

if __name__ == "__main__":

    a = np.loadtxt('data/a.csv', delimiter=',')
    dt = np.loadtxt('data/dt.csv', delimiter=',')
    euler_gt = np.loadtxt('data/euler_gt.csv', delimiter=',')
    gravity = np.loadtxt('data/gravity.csv', delimiter=',')
    omega = np.loadtxt('data/omega.csv', delimiter=',')

    # initialize system
    system = {}
    # gyroscope noise covariance
    system['Q'] = np.array([[0.001, 0, 0, 0, 0, 0],
                        [0, 0.001, 0, 0, 0, 0],
                        [0, 0, 0.001, 0, 0, 0],
                        [0, 0, 0, 0.001, 0, 0],
                        [0, 0, 0, 0, 0.001, 0],
                        [0, 0, 0, 0, 0, 0.001]])
    # accelerometer noise covariance
    system['W'] = np.array([[0.01, 0, 0],
                        [0, 0.01, 0],
                        [0, 0, 0.01]])
    system['gravity'] = gravity

    # build a 2D robot
    robot = {}
    robot['dt'] = dt
    robot['a'] = a
    robot['omega'] = omega
    robot['R'] = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    robot['b'] = np.array([0,0,0])
    robot['P'] = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
    robot['H'] = np.zeros((3,6))
    
    # init imperfect right inekf
    inekf_filter = imperfect_Right_IEKF(system, robot)

    r = R.from_matrix(inekf_filter.R)
    euler_raw = r.as_euler('zyx', degrees=False)
    euler_processed = np.array([[wrapToPI(euler_raw[0]), wrapToPI(euler_raw[1]), wrapToPI(euler_raw[2])]])
    Euler = np.array(euler_processed)
    Bias =  np.array([inekf_filter.b])

    # tracking
    for i in range(len(a)):
        inekf_filter.propagation(i)
        inekf_filter.update(i)

        r = R.from_matrix(inekf_filter.R)
        euler_raw = r.as_euler('zyx', degrees=False)
        euler_processed = np.array([[wrapToPI(euler_raw[0]), wrapToPI(euler_raw[1]), wrapToPI(euler_raw[2])]])
        Euler = np.concatenate((Euler, euler_processed), axis=0)
        bias = np.array([[inekf_filter.b[0], inekf_filter.b[1], inekf_filter.b[2]]])
        Bias = np.concatenate((Bias, bias), axis=0)
        a = 0

    time_stamp = np.empty(1278)
    time_stamp[0] = 0
    for i in range(1, len(dt)+1):
      time_stamp[i] = dt[i-1] + time_stamp[i-1]

    # time_stamp = np.linspace(0, 1277, num=1278)

    fig, axs = plt.subplots(3, figsize=(12,7))
    axs[0].plot(time_stamp, Euler[:, 0], label="x", color='orange')
    axs[0].plot(time_stamp, euler_gt[:, 0], label="x_gt", color='darkgoldenrod')
    axs[0].set_xlabel('time step')
    axs[0].set_ylabel('Euler_x')

    axs[1].plot(time_stamp, Euler[:, 1], label="y", color='lightsalmon')
    axs[1].plot(time_stamp, euler_gt[:, 1], label="y_gt", color='orangered')
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('Euler_y')

    axs[2].plot(time_stamp, Euler[:, 2], label="z", color='slateblue')
    axs[2].plot(time_stamp, euler_gt[:, 2], label="z_gt", color='dodgerblue')
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('Euler_z')


    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig1, axs1 = plt.subplots(3, figsize=(12,7))
    axs1[0].plot(time_stamp, Bias[:, 0], label="x", color='orange')
    axs1[0].set_xlabel('time step')
    axs1[0].set_ylabel('Bias_x')
    axs1[1].plot(time_stamp, Bias[:, 1], label="y", color='lightsalmon')
    axs1[1].set_xlabel('time step')
    axs1[1].set_ylabel('Bias_y')
    axs1[2].plot(time_stamp, Bias[:, 2], label="z", color='slateblue')
    axs1[2].set_xlabel('time step')
    axs1[2].set_ylabel('Bias_z')

    axs1[0].legend()
    axs1[1].legend()
    axs1[2].legend()

    plt.grid()
    plt.show()


