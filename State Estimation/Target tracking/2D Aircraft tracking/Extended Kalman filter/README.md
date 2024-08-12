# Extended Kalman filter (EKF)
Here is an implementation of Extended Kalman filter in a 2-D target tracking scenario using range-bearing measurements. EKF is an extension of Kalman filter with the advantage that this algorithm even works when the motion model and measurement model are non-linear. To test the implementation, please run 'ekf_single_target.py'.

# Problem description
![Screenshot](img/problem_description.jpg)

# Algorithm used
![Screenshot](img/ekf_algo.jpg)

# Results
![Screenshot](img/ekf.png)
- RMSE for x: 0.09045129810577475
- RMSE for y: 0.08108466865609051