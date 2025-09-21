/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_LOOP_H
#define OV_MSCKF_UPDATER_LOOP_H

#include <Eigen/Eigen>
#include <memory>
#include <vector>

#include "UpdaterOptions.h"
#include "state/LoopTypes.h"

namespace ov_msckf {

class State;

/**
 * @brief Tightly-coupled loop closure updater for EKF integration
 *
 * This class provides tightly-coupled loop closure updates by integrating
 * loop constraints directly into the EKF measurement update process.
 * Unlike post-processing approaches, this maintains proper covariance
 * propagation and uncertainty quantification within the MSCKF framework.
 *
 * Key features:
 * - Direct EKF measurement updates for loop constraints
 * - Multi-hypothesis loop closure handling
 * - Incremental covariance updates for computational efficiency
 * - Outlier rejection and robustification
 * - Sparse Jacobian computation for large state vectors
 */
class UpdaterLoop {

public:
  /**
   * @brief Constructor with configuration options
   * @param options Updater options for loop closure
   */
  explicit UpdaterLoop(const UpdaterOptions& options);

  /**
   * @brief Perform tightly-coupled loop closure update
   * @param state Pointer to the current state
   * @param loop_constraints Vector of loop closure constraints
   * @return Number of constraints successfully processed
   */
  int update(std::shared_ptr<State> state,
             const std::vector<LoopConstraint>& loop_constraints);

  /**
   * @brief Update state with a single loop constraint
   * @param state Pointer to the current state
   * @param constraint Loop closure constraint
   * @return true if update successful, false otherwise
   */
  bool updateSingleConstraint(std::shared_ptr<State> state,
                             const LoopConstraint& constraint);

  /**
   * @brief Get number of loop closures processed
   * @return Total number of processed loop closures
   */
  int getNumProcessedLoops() const { return num_processed_loops_; }

  /**
   * @brief Get average processing time per loop closure
   * @return Average processing time in milliseconds
   */
  double getAverageProcessingTime() const {
    return num_processed_loops_ > 0 ? total_processing_time_ / num_processed_loops_ : 0.0;
  }

private:
  /// Options for the updater
  UpdaterOptions options_;

  /// Chi-squared test table for outlier rejection
  std::map<int, double> chi_squared_table_;

  /// Statistics
  int num_processed_loops_;
  double total_processing_time_;

  // Cached matrices for efficiency (avoid repeated allocations)
  Eigen::MatrixXd H_cached_;
  Eigen::VectorXd residual_cached_;
  Eigen::MatrixXd S_cached_;
  Eigen::MatrixXd K_cached_;

  /**
   * @brief Compute measurement residuals and Jacobians for loop constraint
   * @param state Pointer to the current state
   * @param constraint Loop closure constraint
   * @param residual Output residual vector
   * @param H_x Output Jacobian matrix
   * @return true if computation successful, false otherwise
   */
  bool computeResidualAndJacobian(std::shared_ptr<State> state,
                                 const LoopConstraint& constraint,
                                 Eigen::VectorXd& residual,
                                 Eigen::MatrixXd& H_x);

  /**
   * @brief Perform EKF measurement update with loop constraint
   * @param state Pointer to the current state
   * @param residual Measurement residual
   * @param H_x Measurement Jacobian
   * @param R Measurement noise covariance
   * @return true if update successful, false otherwise
   */
  bool performMeasurementUpdate(std::shared_ptr<State> state,
                               const Eigen::VectorXd& residual,
                               const Eigen::MatrixXd& H_x,
                               const Eigen::MatrixXd& R);

  /**
   * @brief Compute pose error between two poses
   * @param pose1 First pose (4x4 matrix)
   * @param pose2 Second pose (4x4 matrix)
   * @param error Output 6D pose error [translation, rotation]
   */
  void computePoseError(const Eigen::Matrix4d& pose1,
                       const Eigen::Matrix4d& pose2,
                       Eigen::VectorXd& error);

  /**
   * @brief Compute Jacobian for pose error
   * @param pose Pose at which to compute Jacobian
   * @param J_pose Output 6x6 Jacobian matrix
   */
  void computePoseJacobian(const Eigen::Matrix4d& pose, Eigen::MatrixXd& J_pose);

  /**
   * @brief Check if constraint passes chi-squared test
   * @param residual Measurement residual
   * @param S Innovation covariance
   * @return true if constraint is an inlier, false if outlier
   */
  bool chiSquaredTest(const Eigen::VectorXd& residual, const Eigen::MatrixXd& S);

  /**
   * @brief Convert JPL quaternion to rotation matrix
   * @param quat JPL quaternion [qx, qy, qz, qw]
   * @return 3x3 rotation matrix
   */
  Eigen::Matrix3d quatToRotMatrix(const Eigen::Vector4d& quat);

  /**
   * @brief Convert rotation matrix to JPL quaternion
   * @param R 3x3 rotation matrix
   * @return JPL quaternion [qx, qy, qz, qw]
   */
  Eigen::Vector4d rotMatrixToQuat(const Eigen::Matrix3d& R);

  /**
   * @brief Compute skew-symmetric matrix for cross product
   * @param vec 3D vector
   * @return 3x3 skew-symmetric matrix
   */
  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec);

  /**
   * @brief Safely invert a matrix with numerical checks
   * @param A Input matrix
   * @param A_inv Output inverted matrix
   * @return true if inversion successful, false otherwise
   */
  bool safeMatrixInvert(const Eigen::MatrixXd& A, Eigen::MatrixXd& A_inv);

  /**
   * @brief Initialize chi-squared table for outlier detection
   */
  void initializeChiSquaredTable();

  /**
   * @brief Get measurement noise covariance for loop constraint
   * @param constraint Loop closure constraint
   * @param R Output measurement noise covariance
   */
  void getMeasurementNoise(const LoopConstraint& constraint, Eigen::MatrixXd& R);
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_LOOP_H