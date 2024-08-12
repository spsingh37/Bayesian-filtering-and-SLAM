
import numpy as np
from scipy.linalg import expm


class imperfect_Right_IEKF:
    def __init__(self, system, robot):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        #   robot:      robot parameters
        # Note that measurement error matrix is a constant for this problem, cuz gravity duhh
        self.Q = system['Q']  # input noise covariance
        self.W = system['W']  # measurement noise covariance
        self.gravity = system['gravity']
        self.dt = robot['dt'] # time stamp
        self.a = robot['a'] # accelerometer measurements
        self.omega = robot['omega'] # angular velocity
        self.R = robot['R'] # robot pose
        self.b = robot['b'] # robot bias
        self.P = robot['P'] # covariance
        self.H = robot['H'] # measurement H matrix

    def skew(self,x):
        # This is useful for the rotation matrix
        """
        :param x: Vector
        :return: R in skew form (NOT R_transpose)
        """
        # vector to skew R^3 -> so(3)
        matrix = np.array([[0, -x[2], x[1]],
                            [x[2], 0, -x[0]],
                            [-x[1], x[0], 0]], dtype=float)
        return matrix

    def propagation(self, i):

        omega_wedge = self.skew(self.omega[i])

        R = self.R @ expm((omega_wedge) * self.dt[i])
        
        A = np.zeros((6,6))
        Phi = expm(A * self.dt[i]) 

        Tmp = np.zeros((6,6))
        Tmp[0:3,0:3] = self.R
        Tmp[3:6,3:6] = np.eye(3)
        Q_bar = Tmp @ self.Q @ Tmp.T
        P = Phi @ self.P @ Phi.T + Phi @ Q_bar @ Phi.T * self.dt[i]

        self.R = R
        self.P = P

    def update(self, i):

        gravity_wedge = self.skew(self.gravity)
        
        self.H[0:3,0:3] = self.R.T @ gravity_wedge
        self.H[0:3,3:6] = np.eye(3)
        N = self.R @ self.W @ self.R.T
        S = self.H @ self.P @ self.H.T + N
        K = self.P @ self.H.T @ np.linalg.inv(S)

        e = K @ (self.a[i] - self.R.T @ self.gravity - self.b)

        temp_wedge = self.skew(e[0:3])
        R = expm(temp_wedge) @ self.R

        b = self.b + e[3:6]
        P = (np.eye(6) - K @ self.H) @ self.P @ (np.eye(6) - K @ self.H).T + K @ N @ K.T

        self.R = R
        self.b = b
        self.P = P
