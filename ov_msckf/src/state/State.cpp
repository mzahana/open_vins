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

#include "State.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

State::State(StateOptions &options) {

  // Save our options
  _options = options;

  // Append the imu to the state and covariance
  int current_id = 0;
  _imu = std::make_shared<IMU>();
  _imu->set_local_id(current_id);
  _variables.push_back(_imu);
  current_id += _imu->size();

  // Append the imu intrinsics to the state and covariance
  // NOTE: these need to be right "next" to the IMU state in the covariance
  // NOTE: since if calibrating these will evolve / be correlated during propagation
  _calib_imu_dw = std::make_shared<Vec>(6);
  _calib_imu_da = std::make_shared<Vec>(6);
  if (options.imu_model == StateOptions::ImuModel::KALIBR) {
    // lower triangular of the matrix (column-wise)
    Eigen::Matrix<double, 6, 1> _imu_default = Eigen::Matrix<double, 6, 1>::Zero();
    _imu_default << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
    _calib_imu_dw->set_value(_imu_default);
    _calib_imu_dw->set_fej(_imu_default);
    _calib_imu_da->set_value(_imu_default);
    _calib_imu_da->set_fej(_imu_default);
  } else {
    // upper triangular of the matrix (column-wise)
    Eigen::Matrix<double, 6, 1> _imu_default = Eigen::Matrix<double, 6, 1>::Zero();
    _imu_default << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
    _calib_imu_dw->set_value(_imu_default);
    _calib_imu_dw->set_fej(_imu_default);
    _calib_imu_da->set_value(_imu_default);
    _calib_imu_da->set_fej(_imu_default);
  }
  _calib_imu_tg = std::make_shared<Vec>(9);
  _calib_imu_GYROtoIMU = std::make_shared<JPLQuat>();
  _calib_imu_ACCtoIMU = std::make_shared<JPLQuat>();
  if (options.do_calib_imu_intrinsics) {

    // Gyroscope dw
    _calib_imu_dw->set_local_id(current_id);
    _variables.push_back(_calib_imu_dw);
    current_id += _calib_imu_dw->size();

    // Accelerometer da
    _calib_imu_da->set_local_id(current_id);
    _variables.push_back(_calib_imu_da);
    current_id += _calib_imu_da->size();

    // Gyroscope gravity sensitivity
    if (options.do_calib_imu_g_sensitivity) {
      _calib_imu_tg->set_local_id(current_id);
      _variables.push_back(_calib_imu_tg);
      current_id += _calib_imu_tg->size();
    }

    // If kalibr model, R_GYROtoIMU is calibrated
    // If rpng model, R_ACCtoIMU is calibrated
    if (options.imu_model == StateOptions::ImuModel::KALIBR) {
      _calib_imu_GYROtoIMU->set_local_id(current_id);
      _variables.push_back(_calib_imu_GYROtoIMU);
      current_id += _calib_imu_GYROtoIMU->size();
    } else {
      _calib_imu_ACCtoIMU->set_local_id(current_id);
      _variables.push_back(_calib_imu_ACCtoIMU);
      current_id += _calib_imu_ACCtoIMU->size();
    }
  }

  // Camera to IMU time offset
  _calib_dt_CAMtoIMU = std::make_shared<Vec>(1);
  if (_options.do_calib_camera_timeoffset) {
    _calib_dt_CAMtoIMU->set_local_id(current_id);
    _variables.push_back(_calib_dt_CAMtoIMU);
    current_id += _calib_dt_CAMtoIMU->size();
  }

  // Loop through each camera and create extrinsic and intrinsics
  for (int i = 0; i < _options.num_cameras; i++) {

    // Allocate extrinsic transform
    auto pose = std::make_shared<PoseJPL>();

    // Allocate intrinsics for this camera
    auto intrin = std::make_shared<Vec>(8);

    // Add these to the corresponding maps
    _calib_IMUtoCAM.insert({i, pose});
    _cam_intrinsics.insert({i, intrin});

    // If calibrating camera-imu pose, add to variables
    if (_options.do_calib_camera_pose) {
      pose->set_local_id(current_id);
      _variables.push_back(pose);
      current_id += pose->size();
    }

    // If calibrating camera intrinsics, add to variables
    if (_options.do_calib_camera_intrinsics) {
      intrin->set_local_id(current_id);
      _variables.push_back(intrin);
      current_id += intrin->size();
    }
  }

  // Finally initialize our covariance to small value
  _Cov = std::pow(1e-3, 2) * Eigen::MatrixXd::Identity(current_id, current_id);

  // Finally, set some of our priors for our calibration parameters
  if (_options.do_calib_imu_intrinsics) {
    _Cov.block(_calib_imu_dw->id(), _calib_imu_dw->id(), 6, 6) = std::pow(0.005, 2) * Eigen::Matrix<double, 6, 6>::Identity();
    _Cov.block(_calib_imu_da->id(), _calib_imu_da->id(), 6, 6) = std::pow(0.008, 2) * Eigen::Matrix<double, 6, 6>::Identity();
    if (_options.do_calib_imu_g_sensitivity) {
      _Cov.block(_calib_imu_tg->id(), _calib_imu_tg->id(), 9, 9) = std::pow(0.005, 2) * Eigen::Matrix<double, 9, 9>::Identity();
    }
    if (_options.imu_model == StateOptions::ImuModel::KALIBR) {
      _Cov.block(_calib_imu_GYROtoIMU->id(), _calib_imu_GYROtoIMU->id(), 3, 3) = std::pow(0.005, 2) * Eigen::Matrix3d::Identity();
    } else {
      _Cov.block(_calib_imu_ACCtoIMU->id(), _calib_imu_ACCtoIMU->id(), 3, 3) = std::pow(0.005, 2) * Eigen::Matrix3d::Identity();
    }
  }
  if (_options.do_calib_camera_timeoffset) {
    _Cov(_calib_dt_CAMtoIMU->id(), _calib_dt_CAMtoIMU->id()) = std::pow(0.01, 2);
  }
  if (_options.do_calib_camera_pose) {
    for (int i = 0; i < _options.num_cameras; i++) {
      _Cov.block(_calib_IMUtoCAM.at(i)->id(), _calib_IMUtoCAM.at(i)->id(), 3, 3) = std::pow(0.005, 2) * Eigen::MatrixXd::Identity(3, 3);
      _Cov.block(_calib_IMUtoCAM.at(i)->id() + 3, _calib_IMUtoCAM.at(i)->id() + 3, 3, 3) =
          std::pow(0.015, 2) * Eigen::MatrixXd::Identity(3, 3);
    }
  }
  if (_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _options.num_cameras; i++) {
      _Cov.block(_cam_intrinsics.at(i)->id(), _cam_intrinsics.at(i)->id(), 4, 4) = std::pow(1.0, 2) * Eigen::MatrixXd::Identity(4, 4);
      _Cov.block(_cam_intrinsics.at(i)->id() + 4, _cam_intrinsics.at(i)->id() + 4, 4, 4) =
          std::pow(0.005, 2) * Eigen::MatrixXd::Identity(4, 4);
    }
  }
}

//===============================================================================
// LOOP CLOSURE IMPLEMENTATION
//===============================================================================

void State::addLoopKeyframe(double timestamp, std::shared_ptr<ov_type::PoseJPL> pose_clone, const KeyframeInfo& keyframe_info) {
  std::lock_guard<std::mutex> lock(_mutex_state);

  // Add keyframe pose
  _keyframes_LOOP[timestamp] = pose_clone;

  // Add keyframe info
  _keyframe_info[timestamp] = keyframe_info;

  // Clean up old keyframes if we exceed the limit
  if (_keyframes_LOOP.size() > static_cast<size_t>(_max_loop_keyframes)) {
    cleanupOldKeyframes();
  }
}

void State::cleanupOldKeyframes() {
  // Remove oldest keyframes to maintain memory limit
  const size_t target_size = static_cast<size_t>(_max_loop_keyframes * 0.8); // Keep 80% when cleaning up

  if (_keyframes_LOOP.size() <= target_size) {
    return;
  }

  // Get timestamps to remove (oldest first)
  std::vector<double> timestamps_to_remove;
  auto it = _keyframes_LOOP.begin();
  size_t num_to_remove = _keyframes_LOOP.size() - target_size;

  for (size_t i = 0; i < num_to_remove && it != _keyframes_LOOP.end(); ++i, ++it) {
    timestamps_to_remove.push_back(it->first);
  }

  // Remove from both maps
  for (double timestamp : timestamps_to_remove) {
    _keyframes_LOOP.erase(timestamp);
    _keyframe_info.erase(timestamp);
  }
}

std::shared_ptr<ov_type::PoseJPL> State::getLoopKeyframePose(double timestamp) {
  std::lock_guard<std::mutex> lock(_mutex_state);

  auto it = _keyframes_LOOP.find(timestamp);
  if (it != _keyframes_LOOP.end()) {
    return it->second;
  }

  // Try to find closest timestamp within tolerance
  const double tolerance = 0.1; // 100ms tolerance
  for (const auto& keyframe_pair : _keyframes_LOOP) {
    if (std::abs(keyframe_pair.first - timestamp) < tolerance) {
      return keyframe_pair.second;
    }
  }

  return nullptr;
}

void State::addLoopConstraint(const LoopConstraint& constraint) {
  std::lock_guard<std::mutex> lock(_mutex_state);
  _loop_constraints.push_back(constraint);
}

void State::removeLoopConstraints(const std::vector<int>& constraint_ids) {
  std::lock_guard<std::mutex> lock(_mutex_state);

  if (constraint_ids.empty()) {
    return;
  }

  // Remove constraints with matching IDs
  auto new_end = std::remove_if(_loop_constraints.begin(), _loop_constraints.end(),
    [&constraint_ids](const LoopConstraint& constraint) {
      return std::find(constraint_ids.begin(), constraint_ids.end(), constraint.constraint_id) != constraint_ids.end();
    });

  _loop_constraints.erase(new_end, _loop_constraints.end());
}
