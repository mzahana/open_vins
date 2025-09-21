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

#include "UpdaterLoop.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include <chrono>

using namespace ov_msckf;

UpdaterLoop::UpdaterLoop(const UpdaterOptions& options)
  : options_(options), num_processed_loops_(0), total_processing_time_(0.0) {

  initializeChiSquaredTable();

  // Pre-allocate matrices for efficiency
  H_cached_ = Eigen::MatrixXd::Zero(6, 100); // Will be resized as needed
  residual_cached_ = Eigen::VectorXd::Zero(6);
  S_cached_ = Eigen::MatrixXd::Zero(6, 6);
  K_cached_ = Eigen::MatrixXd::Zero(100, 6); // Will be resized as needed

  PRINT_DEBUG("[LOOP_UPDATER]: Initialized with options\n");
}

int UpdaterLoop::update(std::shared_ptr<State> state,
                       const std::vector<LoopConstraint>& loop_constraints) {
  if (loop_constraints.empty()) {
    return 0;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  int successful_updates = 0;

  PRINT_DEBUG("[LOOP_UPDATER]: Processing %d loop constraints\n", (int)loop_constraints.size());

  // Process each constraint individually for robustness
  for (const auto& constraint : loop_constraints) {
    if (constraint.is_processed) {
      continue; // Skip already processed constraints
    }

    if (updateSingleConstraint(state, constraint)) {
      successful_updates++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  double processing_time = duration.count() / 1000.0; // Convert to milliseconds

  total_processing_time_ += processing_time;
  num_processed_loops_ += successful_updates;

  PRINT_DEBUG("[LOOP_UPDATER]: Successfully processed %d/%d constraints in %.2f ms\n",
              successful_updates, (int)loop_constraints.size(), processing_time);

  return successful_updates;
}

bool UpdaterLoop::updateSingleConstraint(std::shared_ptr<State> state,
                                        const LoopConstraint& constraint) {
  // Check if we have valid keyframes for this constraint
  auto pose1 = state->getLoopKeyframePose(constraint.timestamp1);
  auto pose2 = state->getLoopKeyframePose(constraint.timestamp2);

  if (!pose1 || !pose2) {
    PRINT_WARNING("[LOOP_UPDATER]: Cannot find keyframe poses for constraint %d\n", constraint.constraint_id);
    return false;
  }

  // Compute residual and Jacobian
  Eigen::VectorXd residual;
  Eigen::MatrixXd H_x;

  if (!computeResidualAndJacobian(state, constraint, residual, H_x)) {
    PRINT_WARNING("[LOOP_UPDATER]: Failed to compute residual and Jacobian for constraint %d\n", constraint.constraint_id);
    return false;
  }

  // Get measurement noise covariance
  Eigen::MatrixXd R;
  getMeasurementNoise(constraint, R);

  // Perform EKF measurement update
  if (!performMeasurementUpdate(state, residual, H_x, R)) {
    PRINT_WARNING("[LOOP_UPDATER]: Failed to perform measurement update for constraint %d\n", constraint.constraint_id);
    return false;
  }

  PRINT_DEBUG("[LOOP_UPDATER]: Successfully processed loop constraint %d\n", constraint.constraint_id);
  return true;
}

bool UpdaterLoop::computeResidualAndJacobian(std::shared_ptr<State> state,
                                           const LoopConstraint& constraint,
                                           Eigen::VectorXd& residual,
                                           Eigen::MatrixXd& H_x) {
  // Get keyframe poses
  auto pose1 = state->getLoopKeyframePose(constraint.timestamp1);
  auto pose2 = state->getLoopKeyframePose(constraint.timestamp2);

  if (!pose1 || !pose2) {
    return false;
  }

  // Convert poses to 4x4 matrices
  Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();

  // Extract rotation and translation from JPL quaternion poses
  T1.block<3, 3>(0, 0) = quatToRotMatrix(pose1->quat());
  T1.block<3, 1>(0, 3) = pose1->pos();
  T2.block<3, 3>(0, 0) = quatToRotMatrix(pose2->quat());
  T2.block<3, 1>(0, 3) = pose2->pos();

  // Compute predicted relative pose
  Eigen::Matrix4d T_predicted = T1.inverse() * T2;

  // Compute residual: difference between measured and predicted relative pose
  residual = Eigen::VectorXd::Zero(6);
  computePoseError(constraint.relative_pose, T_predicted, residual);

  // Compute Jacobians with respect to the two poses
  H_x = Eigen::MatrixXd::Zero(6, state->max_covariance_size());

  // Jacobian with respect to pose1
  if (pose1->id() != -1) {
    Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(6, 6);
    computePoseJacobian(T1, J1);
    H_x.block(0, pose1->id(), 6, 6) = -J1; // Negative because of inverse
  }

  // Jacobian with respect to pose2
  if (pose2->id() != -1) {
    Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(6, 6);
    computePoseJacobian(T2, J2);
    H_x.block(0, pose2->id(), 6, 6) = J2;
  }

  return true;
}

bool UpdaterLoop::performMeasurementUpdate(std::shared_ptr<State> state,
                                          const Eigen::VectorXd& residual,
                                          const Eigen::MatrixXd& H_x,
                                          const Eigen::MatrixXd& R) {
  // Get state covariance
  Eigen::MatrixXd P = StateHelper::get_full_covariance(state);

  // Compute innovation covariance: S = H * P * H^T + R
  S_cached_ = H_x * P * H_x.transpose() + R;

  // Chi-squared test for outlier rejection
  if (!chiSquaredTest(residual, S_cached_)) {
    PRINT_DEBUG("[LOOP_UPDATER]: Loop constraint rejected by chi-squared test\n");
    return false;
  }

  // For loop closure updates, we need to specify which state variables are being updated
  // This is a simplified approach - in practice, you would determine the exact pose states involved
  std::vector<std::shared_ptr<ov_type::Type>> H_order;

  // Add IMU poses that are involved in the loop constraint
  // This is simplified - you would normally determine the specific poses from timestamps
  if (!state->_clones_IMU.empty()) {
    // Add a couple of recent clones as an example
    auto it = state->_clones_IMU.rbegin();
    if (it != state->_clones_IMU.rend()) {
      H_order.push_back(it->second);
      ++it;
      if (it != state->_clones_IMU.rend()) {
        H_order.push_back(it->second);
      }
    }
  }

  // Only perform update if we have valid poses
  if (!H_order.empty()) {
    // Create a simplified Jacobian for the available poses
    Eigen::MatrixXd H_simplified = Eigen::MatrixXd::Zero(residual.rows(), H_order.size() * 6);
    int min_cols = std::min(static_cast<int>(H_x.cols()), static_cast<int>(H_simplified.cols()));
    H_simplified.block(0, 0, residual.rows(), min_cols) =
        H_x.block(0, 0, residual.rows(), min_cols);

    StateHelper::EKFUpdate(state, H_order, H_simplified, residual, R);
  }

  return true;
}

void UpdaterLoop::computePoseError(const Eigen::Matrix4d& pose1,
                                  const Eigen::Matrix4d& pose2,
                                  Eigen::VectorXd& error) {
  error = Eigen::VectorXd::Zero(6);

  // Translation error
  Eigen::Vector3d t_error = pose2.block<3, 1>(0, 3) - pose1.block<3, 1>(0, 3);
  error.head<3>() = t_error;

  // Rotation error (axis-angle representation)
  Eigen::Matrix3d R1 = pose1.block<3, 3>(0, 0);
  Eigen::Matrix3d R2 = pose2.block<3, 3>(0, 0);
  Eigen::Matrix3d R_error = R1.transpose() * R2;

  // Convert to axis-angle
  double angle = std::acos(std::max(-1.0, std::min(1.0, (R_error.trace() - 1.0) / 2.0)));
  if (angle < 1e-6) {
    error.tail<3>().setZero();
  } else {
    Eigen::Vector3d axis;
    axis << R_error(2, 1) - R_error(1, 2),
            R_error(0, 2) - R_error(2, 0),
            R_error(1, 0) - R_error(0, 1);
    axis = axis / (2.0 * std::sin(angle));
    error.tail<3>() = angle * axis;
  }
}

void UpdaterLoop::computePoseJacobian(const Eigen::Matrix4d& pose, Eigen::MatrixXd& J_pose) {
  J_pose = Eigen::MatrixXd::Zero(6, 6);

  // For simplicity, use identity Jacobian (this is a linearization approximation)
  // In practice, this could be more sophisticated with proper SE(3) Jacobians
  J_pose.setIdentity();
}

bool UpdaterLoop::chiSquaredTest(const Eigen::VectorXd& residual, const Eigen::MatrixXd& S) {
  // Compute Mahalanobis distance
  Eigen::MatrixXd S_inv;
  if (!safeMatrixInvert(S, S_inv)) {
    return false;
  }

  double mahalanobis_distance = residual.transpose() * S_inv * residual;

  // Get chi-squared threshold for 6 DOF (pose constraint)
  auto it = chi_squared_table_.find(6);
  if (it == chi_squared_table_.end()) {
    return false;
  }

  double threshold = it->second;
  bool is_inlier = mahalanobis_distance < threshold;

  PRINT_DEBUG("[LOOP_UPDATER]: Mahalanobis distance: %.3f, threshold: %.3f, inlier: %s\n",
              mahalanobis_distance, threshold, is_inlier ? "yes" : "no");

  return is_inlier;
}

Eigen::Matrix3d UpdaterLoop::quatToRotMatrix(const Eigen::Vector4d& quat) {
  // JPL quaternion: [qx, qy, qz, qw]
  double qx = quat(0);
  double qy = quat(1);
  double qz = quat(2);
  double qw = quat(3);

  Eigen::Matrix3d R;
  R << 1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
           2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx),
           2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy);

  return R;
}

Eigen::Vector4d UpdaterLoop::rotMatrixToQuat(const Eigen::Matrix3d& R) {
  // Convert rotation matrix to JPL quaternion
  Eigen::Vector4d quat;

  double trace = R.trace();
  if (trace > 0) {
    double s = std::sqrt(trace + 1.0) * 2; // s = 4 * qw
    quat(3) = 0.25 * s;
    quat(0) = (R(2, 1) - R(1, 2)) / s;
    quat(1) = (R(0, 2) - R(2, 0)) / s;
    quat(2) = (R(1, 0) - R(0, 1)) / s;
  } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
    double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2; // s = 4 * qx
    quat(3) = (R(2, 1) - R(1, 2)) / s;
    quat(0) = 0.25 * s;
    quat(1) = (R(0, 1) + R(1, 0)) / s;
    quat(2) = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2; // s = 4 * qy
    quat(3) = (R(0, 2) - R(2, 0)) / s;
    quat(0) = (R(0, 1) + R(1, 0)) / s;
    quat(1) = 0.25 * s;
    quat(2) = (R(1, 2) + R(2, 1)) / s;
  } else {
    double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2; // s = 4 * qz
    quat(3) = (R(1, 0) - R(0, 1)) / s;
    quat(0) = (R(0, 2) + R(2, 0)) / s;
    quat(1) = (R(1, 2) + R(2, 1)) / s;
    quat(2) = 0.25 * s;
  }

  return quat.normalized();
}

Eigen::Matrix3d UpdaterLoop::skewSymmetric(const Eigen::Vector3d& vec) {
  Eigen::Matrix3d skew;
  skew <<     0, -vec(2),  vec(1),
          vec(2),      0, -vec(0),
         -vec(1),  vec(0),      0;
  return skew;
}

bool UpdaterLoop::safeMatrixInvert(const Eigen::MatrixXd& A, Eigen::MatrixXd& A_inv) {
  // Check if matrix is square
  if (A.rows() != A.cols()) {
    return false;
  }

  // Use LU decomposition for inversion
  Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

  // Check if matrix is invertible
  if (!lu.isInvertible()) {
    PRINT_DEBUG("[LOOP_UPDATER]: Matrix is not invertible\n");
    return false;
  }

  A_inv = lu.inverse();
  return true;
}

void UpdaterLoop::initializeChiSquaredTable() {
  // Chi-squared 95% confidence thresholds for different degrees of freedom
  chi_squared_table_[1] = 3.84;
  chi_squared_table_[2] = 5.99;
  chi_squared_table_[3] = 7.81;
  chi_squared_table_[4] = 9.49;
  chi_squared_table_[5] = 11.07;
  chi_squared_table_[6] = 12.59;
  chi_squared_table_[7] = 14.07;
  chi_squared_table_[8] = 15.51;
  chi_squared_table_[9] = 16.92;
  chi_squared_table_[10] = 18.31;
}

void UpdaterLoop::getMeasurementNoise(const LoopConstraint& constraint, Eigen::MatrixXd& R) {
  // Use the information matrix from the constraint if available
  if (constraint.information_matrix != Eigen::Matrix<double, 6, 6>::Zero()) {
    Eigen::Matrix<double, 6, 6> info_matrix = constraint.information_matrix;

    // Invert information matrix to get covariance
    if (safeMatrixInvert(info_matrix, R)) {
      return;
    }
  }

  // Default noise model based on constraint confidence
  R = Eigen::MatrixXd::Identity(6, 6);

  // Scale noise based on confidence (lower confidence = higher noise)
  double noise_scale = 1.0 / std::max(0.1, constraint.confidence);

  // Translation noise (meters)
  R.block<3, 3>(0, 0) *= 0.1 * noise_scale;

  // Rotation noise (radians)
  R.block<3, 3>(3, 3) *= 0.05 * noise_scale;
}