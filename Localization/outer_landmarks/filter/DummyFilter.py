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
class DummyFilter:

    def __init__(self, system, init):
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # system matrix Jacobian
        self.hfun = system.hfun  # input matrix Jacobian
        self.M = system.M
        self.Q = system.Q

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        X = self.state_.getState()
        P = self.state_.getCovariance()

        X_pred = self.gfun(X, u)
        P_pred = np.eye(3)

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # do nothing correct step.

        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        X = X_predict
        P =  P_predict

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state