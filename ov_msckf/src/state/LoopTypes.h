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

#ifndef OV_MSCKF_STATE_LOOP_TYPES_H
#define OV_MSCKF_STATE_LOOP_TYPES_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ov_msckf {

/**
 * @brief Loop closure constraint structure
 */
struct LoopConstraint {
  int constraint_id;                    // Unique constraint identifier
  double timestamp1;                    // First keyframe timestamp
  double timestamp2;                    // Second keyframe timestamp
  Eigen::Matrix4d relative_pose;        // Relative pose T_2_1 (from frame1 to frame2)
  Eigen::Matrix<double, 6, 6> information_matrix; // Information matrix for constraint
  std::vector<cv::DMatch> inlier_matches;         // Feature matches used
  double confidence;                    // Confidence score [0, 1]
  bool is_processed;                   // Whether constraint has been used in update

  LoopConstraint() : constraint_id(-1), timestamp1(0.0), timestamp2(0.0),
                     relative_pose(Eigen::Matrix4d::Identity()),
                     information_matrix(Eigen::Matrix<double, 6, 6>::Identity()),
                     confidence(0.0), is_processed(false) {}

  LoopConstraint(int id, double t1, double t2, const Eigen::Matrix4d& pose,
                 const Eigen::Matrix<double, 6, 6>& info, double conf)
    : constraint_id(id), timestamp1(t1), timestamp2(t2), relative_pose(pose),
      information_matrix(info), confidence(conf), is_processed(false) {}
};

/**
 * @brief Keyframe information for loop detection
 */
struct KeyframeInfo {
  double timestamp;                     // Keyframe timestamp
  std::vector<cv::KeyPoint> keypoints;  // Detected keypoints
  cv::Mat descriptors;                  // Feature descriptors (for backward compatibility)
  std::vector<size_t> feature_ids;      // Associated feature IDs
  int num_tracked_features;             // Number of successfully tracked features
  bool is_keyframe_selected;            // Whether this was selected as keyframe

  KeyframeInfo() : timestamp(0.0), num_tracked_features(0), is_keyframe_selected(false) {}

  KeyframeInfo(double t, const std::vector<cv::KeyPoint>& kpts,
               const cv::Mat& desc, const std::vector<size_t>& feat_ids)
    : timestamp(t), keypoints(kpts), descriptors(desc), feature_ids(feat_ids),
      num_tracked_features(static_cast<int>(feat_ids.size())), is_keyframe_selected(true) {}
};

/**
 * @brief Loop detection result structure
 */
struct LoopDetectionResult {
  bool loop_detected;                   // Whether a loop was detected
  std::vector<LoopConstraint> constraints; // Detected loop constraints
  double detection_time;                // Time spent on detection (ms)
  int num_candidates_evaluated;        // Number of candidates evaluated

  LoopDetectionResult() : loop_detected(false), detection_time(0.0), num_candidates_evaluated(0) {}
};

// KeyframeSelectionCriteria is defined in VioManager.h to avoid circular dependencies

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_LOOP_TYPES_H