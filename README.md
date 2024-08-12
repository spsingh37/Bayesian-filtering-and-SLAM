## 🤖 Bayesian Filtering & SLAM                                                               
Note:  This project is a culmination of my work for the **"ROB 530: Mobile Robotics: Methods & Algorithms"** course, conducted from January to April 2023. It showcases various methods for state estimation, localization, mapping, and SLAM, essential for ground vehicles, underwater robots, and aerial vehicles.
<!-- <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/acc_vehicle_simulation.gif">
</div> -->
<!-- <div style="display: flex; justify-content: center;">
    <div style="margin: 10px;">
        <img src="assets/acc_vehicle_simulation.gif" alt="Simulation 1" style="width: 200px;">
    </div>
    <div style="margin: 10px;">
        <img src="assets/vehicle_overtaking.gif" alt="Simulation 2" style="width: 200px;">
    </div>
</div> -->
<p align="center">
  <img src="assets/bayes_filter_gif.gif" alt="Bayes filter" width="400" />
  <img src="assets/mapping_gif.gif" alt="Mapping" width="400" />
  <img src="assets/riekf_localization_gif.gif" alt="RIEKF Localization" width="400" />
  <img src="assets/ls_incr_slam_gif.gif" alt="LS Incremental SLAM" width="400" />
</p>

### 🎯 Goal
This repository encompasses several critical components of autonomous robotic systems, including:

- **State estimation**: Implementing target tracking for 1D, 2D, and 3D scenarios using various filters such as Kalman Filter (KF), Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), Particle Filter (PF), and Invariant Extended Kalman Filter (InEKF). Attitude-Heading Reference system using Right Invariant Extended Kalman Filter (R-InEKF). 
- **Mapping**: Implementing algorithms for discrete, continuous 2D occupancy grid both with and without semantic categories.
- **Localization**: Implementing algorihms for robot localization using various filters.
- **SLAM**: Implementing various types of SLAM: 2D EKF-based SLAM, Least Squares SLAM, Incremental Least Squares SLAM, 2D & 3D Pose Graph SLAM.

## ⚙️ Prerequisites
- Python libraries:
    - Numpy
    - Matplotlib
    - Scipy
    - ROS Noetic (for one of the cases of Robot localization)
    - gtsam (for Pose Graph SLAM)
    - sksparse (for Batch Least Squares SLAM)

## 🛠️ Test/Demo
- State Estimation
    - Target tracking
        - 1D Circular hallway robot tracking
            - Launch the jupyter notebook containing Bayes filter implementation
        - 2D Aircraft tracking
            - KF: Run the 'kf_single_target.py'
            - EKF: Run the 'ekf_single_target.py'
            - UKF: Run the 'ukf_single_target.py'
            - PF: Run the 'pf_single_target.py'
        - 3D Camera-based tracking
            - Launch the jupyter notebook containing EKF and PF implementations (batch & incremental)
    - Attitude-Heading Reference System
            - Without bias in accelerometer: Run 'without_bias_riekf.py'
            - With bias in accelerometer: Run 'with_bias_imperfect_riekf.py'
- Mapping
    - Run 'run.py' with argument (1-4) to choose discrete or continuous 2D occupancy grid mapping, and whether semantic or not
- Localization
    - inner_landmarks
        - Run 'riekf_localization_se2.py' or launch the notebook
    - outer_landmarks:
        - Run 'run.py' and other instructions for visualization and which filter to use mentioned in the README inside the directory
- SLAM
    - 2D EKF and Least-Sq. SLAM (Batch & Incremental)
        - EKF SLAM: Run 'ekf_slam.py'
        - Least Squares SLAM (batch): Run 'ls_slam.py'
        - Incremental Least Squares SLAM: Run 'ls_incremental_slam.py'
    - 2D & 3D Pose Graph SLAM
        - Run 'main.py' with appropriate arguments specifying Gauss-Newton SLAM or Incremental SLAM, and whether 2D or 3D SLAM

## 📊 Results
### 📈 State Estimation
#### Target tracking
- 1D Bayesian filtering for tracking robot position in a circular hallway
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/bayes_filter_gif.gif">
</div>

<!-- - 2D Aircraft tracking
<p align="center">
  <img src="assets/kf_results.png" alt="KF" width="400" />
  <img src="assets/ekf_results.png" alt="EKF" width="400" />
  <img src="assets/ukf_results.png" alt="UKF" width="400" />
  <img src="assets/pf_results.png" alt="PF" width="400" />
</p> -->
- 2D Aircraft tracking
<table align="center">
  <tr>
    <td align="center">
      <figcaption>Kalman Filter (KF)</figcaption>
      <img src="assets/kf_results.png" alt="KF" width="400" />
    </td>
    <td align="center">
      <figcaption>Extended Kalman Filter (EKF)</figcaption>
      <img src="assets/ekf_results.png" alt="EKF" width="400" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <figcaption>Unscented Kalman Filter (UKF)</figcaption>
      <img src="assets/ukf_results.png" alt="UKF" width="400" />
    </td>
    <td align="center">
      <figcaption>Particle Filter (PF)</figcaption>
      <img src="assets/pf_results.png" alt="PF" width="400" />
    </td>
  </tr>
</table>


#### Camera-based tracking of an object in 3D space 
<p align="center">
  <img src="assets/ekf_sequential.png" alt="EKF_seq" width="400" />
  <img src="assets/ekf_batch.png" alt="EKF_batch" width="400" />
  <img src="assets/pf_sequential.png" alt="PF_seq" width="400" />
  <img src="assets/pf_batch.png" alt="PF_batch" width="400" />
</p>

### 📈 Mapping
<p align="center">
  <img src="assets/csm_mean_variance.jpg" alt="CSM" width="400" />
  <img src="assets/s-csm_mean_variance.jpg" alt="S-CSM" width="400" />
  <img src="assets/cont_csm_mean_variance.jpg" alt="Cont_CSM" width="400" />
  <img src="assets/cont_s-csm_mean_variance.jpg" alt="Cont_S-CSM" width="400" />
</p>

### 📈 Robot Localization
#### With landmarks inside the field
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/riekf_localization_gif.gif">
</div>

#### With landmarks outside the field
<p align="center">
  <img src="assets/ekf_localization_gif.gif" alt="EKF" width="400" />
  <img src="assets/ukf_localization_gif.gif" alt="UKF" width="400" />
  <img src="assets/pf_localization_gif.gif" alt="PF" width="400" />
  <img src="assets/inekf_localization_gif.gif" alt="InEKF" width="400" />
</p>

### 📈 Simultaneous Localization & Mapping (SLAM)
#### EKF-based SLAM, and Least Squares SLAM:
- EKF-based SLAM
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/ekf_slam_gif.gif">
</div>

- Least Squares SLAM:

    - Batch
    <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/ls_slam.png">
    </div>

    - Incremental
    <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/ls_incr_slam_gif.gif">
    </div>

#### 2D and 3D Pose Graph SLAM:
- 2D Batch Gauss-Newton Graph SLAM (Intel dataset)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/gn_2d.png">
</div>

- 2D Incremental Least Squares Graph SLAM (Intel dataset)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/isam2_2d_gif.gif">
</div>

- 3D Batch Gauss-Newton Graph SLAM (Parking garage dataset)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/gn_3d.png">
</div>

- 3D Incremental Least Squares Graph SLAM (Parking garage dataset)
<div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/isam2_3d_gif.gif">
</div>