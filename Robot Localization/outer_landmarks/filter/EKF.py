import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# init.mu = initialStateMean
#         init.Sigma = initialStateCov
#         init.Gfun = Gfun
#         init.Vfun = Vfun
#         init.Hfun = Hfun
#         from filter.EKF import EKF
#         filter = EKF(sys, init)

# sys.gfun = gfun
#     sys.hfun = hfun
#     sys.M = partial(M, alphas=alphas)
#     sys.Q = np.array([[beta**2,0],[0,25**2]])
#     sys.W = np.diag([0.05**2,0.1**2,0.1**2])
#     sys.V = 10000*np.eye(2)

# Extended Kalman filter class for state estimation of a nonlinear system
class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # TODO: add comments for each attribute
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # system matrix Jacobian
        self.hfun = system.hfun  # input matrix Jacobian
        self.Gfun = init.Gfun  # process model
        self.Vfun = init.Vfun  # measurement model Jacobian
        self.Hfun = init.Hfun  # input noise covariance
        self.M = system.M
        self.Q = system.Q

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):
        X = self.state_.getState()
        P = self.state_.getCovariance()

        # Evaluate G with mean and input
        G = self.Gfun(X, u)
        # Evaluate V with mean and input
        V = self.Vfun(X, u)
        # Propoagate mean through non-linear dynamics
        X_pred = self.gfun(X, u)
        # Update covariance with G,V and M(u)
        P_pred = G @ P @ G.T + V @ self.M(u) @ V.T

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        z_hat1 = self.hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict)
        z_hat2 = self.hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict)
        z_hat = np.hstack((z_hat1,z_hat2))

        # evaluate measurement Jacobian at current operating point
        H_1 = self.Hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict,z_hat1)
        H_2 = self.Hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict,z_hat2)
        H = np.vstack((H_1,H_2))
        
        # compute innovation statistics
        # We know here z[1] is an angle
        z_no_id = np.hstack((z[0:2],z[3:5]))
        v = z_no_id - z_hat  # innovation

        S = H @ P_predict @ H.T + block_diag(self.Q,self.Q) # innovation covariance
      
        K = P_predict @ H.T @ np.linalg.inv(S)  # Kalman (filter) gain

        # correct the predicted state statistics
        # TODO: check this!
        diff = [
                wrap2Pi(z[0] - z_hat1[0]),
                z[1] - z_hat1[1],
                wrap2Pi(z[3] - z_hat2[0]),
                z[4] - z_hat2[1]]
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        U = np.eye(np.shape(X)[0]) - K @ H
        P = U @ P_predict @ U.T + K @ block_diag(self.Q,self.Q) @ K.T

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state