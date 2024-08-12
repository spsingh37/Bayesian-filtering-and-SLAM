
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


# Unscented transform class for uncertainty
# Unscented Kalman filter class for state estimation of a nonlinear system
class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)



    def prediction(self, u):
        # UKF propagation (prediction) step
        mean = self.state_.getState()
        mean_aug = np.hstack((mean, np.zeros(3))).reshape(-1,1)
        sigma = self.state_.getCovariance()
        sigma_aug = np.zeros((6,6))
        sigma_aug[:3,:3] = sigma
        sigma_aug[3:,3:] = self.M(u)
        self.sigma_point(mean_aug, sigma_aug, self.kappa_g)
        X_pred = np.zeros((3,1))
        self.Y = np.zeros((3, 2*self.n+1))
        for j in range(self.Y.shape[1]):
            self.Y[:,j] = self.gfun(self.X[:3,j], u+self.X[3:,j])
            
            X_pred += self.w[j]*self.Y[:,j].reshape(-1,1)
        
        P_pred = (self.Y - X_pred) @ np.diag(self.w) @ (self.Y-X_pred).T
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):

        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        Z = np.zeros((4,2*self.n+1))
        z_hat = np.zeros((4,1))
        for j in range(Z.shape[1]):
            Z[:2,j] = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.Y[:,j])
            Z[2:,j] = self.hfun(landmark2.getPosition()[0],landmark2.getPosition()[1], self.Y[:,j])
            z_hat += self.w[j]*Z[:,j].reshape(-1,1)
        

        S = (Z-z_hat) @ np.diag(self.w) @ (Z-z_hat).T + block_diag(self.Q,self.Q)
        Sigma_xz = (self.Y - X_predict) @ np.diag(self.w) @ (Z - z_hat).T
        K = Sigma_xz @ np.linalg.inv(S)
        diff = [
                wrap2Pi(z[0] - z_hat[0]),
                z[1] - z_hat[1],
                wrap2Pi(z[3] - z_hat[2]),
                z[4] - z_hat[3]]
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        P = P_predict - K @ S @ K.T

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X.reshape(3))
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        # mean = mean.reshape(-1,1)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state