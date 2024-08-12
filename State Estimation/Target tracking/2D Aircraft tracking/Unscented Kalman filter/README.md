# Unscented Kalman filter (UKF)
Here is an implementation of Unscented Kalman filter in a 2-D target tracking scenario using range-bearing measurements. UKF is an extension of Kalman filter, and just like EKF this algorithm also works when the motion model and measurement model are non-linear. This algorithm gives better state approximation than EKF, and on top of that there are no derivatives in UKF. To test the implementation, please run 'ukf_single_target.py'.

# Problem description
![Screenshot](img/problem_description.jpg)

# Algorithm used
![Screenshot](img/ukf_algo.jpg)

# Results
![Screenshot](img/ukf.png)
- RMSE for x: 0.043375151272279006
- RMSE for y: 0.09822741624096383