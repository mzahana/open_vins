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

#ifndef OV_MSCKF_LOOP_TYPES_H
#define OV_MSCKF_LOOP_TYPES_H

#include <Eigen/Eigen>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ov_msckf {

/**
 * @brief BRIEF descriptor type (256-bit binary descriptor for efficiency)
 */
using BriefDescriptor = std::bitset<256>;

/**
 * @brief Keyframe data structure for efficient storage
 */
struct KeyframeData {
  double timestamp;
  Eigen::Matrix4d pose;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<BriefDescriptor> descriptors;

  KeyframeData() : timestamp(0.0), pose(Eigen::Matrix4d::Identity()) {}

  KeyframeData(double t, const Eigen::Matrix4d& p,
               const std::vector<cv::KeyPoint>& kpts,
               const std::vector<BriefDescriptor>& desc)
    : timestamp(t), pose(p), keypoints(kpts), descriptors(desc) {}
};

/**
 * @brief Loop closure candidate structure
 */
struct LoopCandidate {
  int query_keyframe_id;
  int match_keyframe_id;
  double similarity_score;
  double geometric_score;
  double temporal_consistency_score;
  Eigen::Matrix4d relative_pose;
  std::vector<cv::DMatch> inlier_matches;
  bool is_verified;
  double confidence;

  LoopCandidate() : query_keyframe_id(-1), match_keyframe_id(-1),
                    similarity_score(0.0), geometric_score(0.0),
                    temporal_consistency_score(0.0),
                    relative_pose(Eigen::Matrix4d::Identity()),
                    is_verified(false), confidence(0.0) {}
};

/**
 * @brief Loop detection task for threading
 */
struct DetectionTask {
  KeyframeData keyframe;
  double current_timestamp;
  int task_id;

  DetectionTask(const KeyframeData& kf, double t, int id)
    : keyframe(kf), current_timestamp(t), task_id(id) {}
};

/**
 * @brief Options for loop detector configuration
 */
struct LoopDetectorOptions {
  bool enabled = true;
  std::string vocabulary_path = "";

  // Resource management
  double detection_frequency = 1.0; // Hz
  int max_keyframes_memory = 1000;
  int processing_threads = 2;

  // Detection parameters
  double min_loop_separation = 30.0; // seconds
  double bow_similarity_threshold = 0.01;
  int geometric_verification_threshold = 30;
  int temporal_consistency_window = 3;
  int max_candidates_per_detection = 5;

  // Performance adaptation
  bool enable_resource_monitoring = false;
  double cpu_usage_threshold = 80.0;
  double memory_usage_threshold = 85.0;
  bool adaptive_frequency_scaling = true;

  // Advanced parameters
  int brief_patch_size = 48;
  int max_features_per_keyframe = 500;
  double ransac_probability = 0.99;
  int max_ransac_iterations = 1000;
  double essential_matrix_threshold = 1.0;
};

/**
 * @brief Circular buffer template for fixed memory allocation
 */
template<typename T>
class CircularBuffer {
private:
  std::vector<T> buffer_;
  size_t head_;
  size_t tail_;
  size_t max_size_;
  bool full_;

public:
  explicit CircularBuffer(size_t size)
    : buffer_(size), head_(0), tail_(0), max_size_(size), full_(false) {}

  void push(const T& item) {
    buffer_[head_] = item;
    if (full_) {
      tail_ = (tail_ + 1) % max_size_;
    }
    head_ = (head_ + 1) % max_size_;
    full_ = head_ == tail_;
  }

  bool empty() const {
    return (!full_ && (head_ == tail_));
  }

  bool full() const {
    return full_;
  }

  size_t size() const {
    size_t size = max_size_;
    if (!full_) {
      if (head_ >= tail_) {
        size = head_ - tail_;
      } else {
        size = max_size_ + head_ - tail_;
      }
    }
    return size;
  }

  T& operator[](size_t idx) {
    return buffer_[(tail_ + idx) % max_size_];
  }

  const T& operator[](size_t idx) const {
    return buffer_[(tail_ + idx) % max_size_];
  }

  void clear() {
    head_ = tail_ = 0;
    full_ = false;
  }
};

/**
 * @brief Performance metrics for resource monitoring
 */
struct PerformanceMetrics {
  double cpu_usage = 0.0;
  double memory_usage = 0.0;
  double detection_frequency = 0.0;
  double average_detection_time = 0.0;
  int active_keyframes = 0;
  int total_loop_closures = 0;

  void reset() {
    cpu_usage = memory_usage = detection_frequency = average_detection_time = 0.0;
    active_keyframes = total_loop_closures = 0;
  }
};

} // namespace ov_msckf

#endif // OV_MSCKF_LOOP_TYPES_H