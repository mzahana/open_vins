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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "VioManagerOptions.h"

// Forward declarations for loop closure
namespace ov_msckf {
  class LoopDetector;
  class BriefExtractor;

  // Simple keyframe selection criteria (avoiding header dependencies)
  struct KeyframeSelectionCriteria {
    double min_translation_distance = 0.3;   // Minimum translation distance (meters)
    double min_rotation_angle = 0.2;         // Minimum rotation angle (radians)

    bool shouldSelectKeyframe(const Eigen::Matrix4d& last_pose,
                             const Eigen::Matrix4d& current_pose,
                             int num_tracked_features,
                             int total_features) const {
      // Check translation distance
      Eigen::Vector3d translation_diff = current_pose.block<3, 1>(0, 3) - last_pose.block<3, 1>(0, 3);
      double translation_distance = translation_diff.norm();

      if (translation_distance > min_translation_distance) {
        return true;
      }

      // Check rotation angle
      Eigen::Matrix3d R_last = last_pose.block<3, 3>(0, 0);
      Eigen::Matrix3d R_current = current_pose.block<3, 3>(0, 0);
      Eigen::Matrix3d R_diff = R_current * R_last.transpose();

      double rotation_angle = std::acos(std::max(-1.0, std::min(1.0, (R_diff.trace() - 1.0) / 2.0)));

      return rotation_angle > min_rotation_angle;
    }
  };
}

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class UpdaterLoop;
class Propagator;

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// If we are initialized or not
  bool initialized() { return is_initialized_vio && timelastupdate != -1; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image();

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// Get loop closure detection status
  bool is_loop_closure_enabled() const;

  /// Get number of loop closures detected
  int get_num_loop_closures() const;

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /**
   * @brief Perform loop closure detection and update
   * @param message Contains our timestamp, images, and camera ids
   */
  void process_loop_closure(const ov_core::CameraData &message);

  /**
   * @brief Check if current frame should be selected as keyframe
   * @param timestamp Current timestamp
   * @param pose Current camera pose
   * @return true if frame should be keyframe
   */
  bool should_select_keyframe(double timestamp, const Eigen::Matrix4d& pose);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// Our loop closure updater
  std::shared_ptr<UpdaterLoop> updaterLOOP;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;

  // If we did a zero velocity update
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;

  //===============================================================================
  // LOOP CLOSURE VARIABLES
  //===============================================================================

  /// Loop detector instance
  std::shared_ptr<ov_msckf::LoopDetector> loop_detector;

  /// Brief descriptor extractor for loop closure
  std::shared_ptr<ov_msckf::BriefExtractor> brief_extractor;

  /// Last keyframe timestamp for keyframe selection
  double last_keyframe_timestamp = -1;

  /// Last keyframe pose for keyframe selection
  Eigen::Matrix4d last_keyframe_pose = Eigen::Matrix4d::Identity();

  /// Keyframe selection criteria
  ov_msckf::KeyframeSelectionCriteria keyframe_criteria;

  /// Number of loop closures detected
  int num_loop_closures = 0;

  /// Loop closure processing frequency control
  double last_loop_detection_time = -1;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
