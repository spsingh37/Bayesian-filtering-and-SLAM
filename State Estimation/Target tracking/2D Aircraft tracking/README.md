# 2D Aircraft tracking
Here, I present implementations of some fundamental algorithms which are pretty handy in the field of mobile robotics. I learned these algorithms through the NAV 568 course taught at the University of Michigan. Though, here the application scenario is for 2D Aircraft tracking with range & bearing measurements.

## Results overview

### Kalman filter
![Screenshot](Kalman filter/img/kf.png)
- RMSE for x: 0.12045962625789276
- RMSE for y: 0.07947022680561029

### Extended Kalman filter
![Screenshot](Extended Kalman filter/img/ekf.png)
- RMSE for x: 0.09045129810577475
- RMSE for y: 0.08108466865609051

### Unscented Kalman filter
![Screenshot](Unscented Kalman filter/img/ukf.png)
- RMSE for x: 0.043375151272279006
- RMSE for y: 0.09822741624096383

### Particle filter
![Screenshot](Particle filter/img/pf.png)
- RMSE for x: 0.0506806397884717
- RMSE for y: 0.05046825970440439