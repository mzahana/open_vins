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

#include "LoopDetector.h"
#include "utils/print.h"
#include <algorithm>
#include <chrono>
#include <opencv2/features2d.hpp>

using namespace ov_msckf;

LoopDetector::LoopDetector(const LoopDetectorOptions& options)
  : options_(options), is_initialized_(false), is_shutdown_(false),
    keyframe_history_(options.max_keyframes_memory), next_keyframe_id_(0),
    next_task_id_(0) {

  bow_database_ = std::make_unique<BowDatabase>();
  brief_extractor_ = std::make_unique<BriefExtractor>(options.brief_patch_size);

  performance_metrics_.reset();
  last_detection_time_ = std::chrono::steady_clock::now();
}

LoopDetector::~LoopDetector() {
  shutdown();
}

bool LoopDetector::initialize() {
  if (is_initialized_) {
    return true;
  }

  // Initialize BoW database
  if (!bow_database_->initialize(options_.vocabulary_path)) {
    PRINT_ERROR("[LOOP]: Failed to initialize BoW database\n");
    return false;
  }

  // Start detection thread
  is_shutdown_ = false;
  detection_thread_ = std::thread(&LoopDetector::detectionThreadFunc, this);

  is_initialized_ = true;
  PRINT_INFO("[LOOP]: Loop detector initialized successfully\n");
  return true;
}

void LoopDetector::shutdown() {
  if (!is_initialized_ || is_shutdown_) {
    return;
  }

  is_shutdown_ = true;

  // Notify detection thread to stop
  {
    std::lock_guard<std::mutex> lock(detection_mutex_);
    detection_cv_.notify_all();
  }

  // Wait for thread to finish
  if (detection_thread_.joinable()) {
    detection_thread_.join();
  }

  PRINT_INFO("[LOOP]: Loop detector shutdown complete\n");
}

bool LoopDetector::addKeyframe(double timestamp,
                              const std::vector<cv::KeyPoint>& keypoints,
                              const std::vector<BriefDescriptor>& descriptors,
                              const Eigen::Matrix4d& pose) {
  PRINT_DEBUG("[LOOP_DETECTOR]: addKeyframe called with timestamp %.3f, %zu keypoints, %zu descriptors\n",
              timestamp, keypoints.size(), descriptors.size());

  if (!is_initialized_ || is_shutdown_) {
    PRINT_DEBUG("[LOOP_DETECTOR]: addKeyframe failed - not initialized (%s) or shutdown (%s)\n",
                is_initialized_ ? "true" : "false", is_shutdown_ ? "true" : "false");
    return false;
  }

  // Validate input data
  if (keypoints.empty() || descriptors.empty() || keypoints.size() != descriptors.size()) {
    PRINT_DEBUG("[LOOP_DETECTOR]: addKeyframe failed - invalid input data (keypoints: %zu, descriptors: %zu)\n",
                keypoints.size(), descriptors.size());
    return false;
  }

  // Create keyframe data
  KeyframeData keyframe(timestamp, pose, keypoints, descriptors);

  // Add to circular buffer and map
  keyframe_history_.push(keyframe);
  int keyframe_id = next_keyframe_id_++;
  keyframe_map_[keyframe_id] = keyframe;

  PRINT_DEBUG("[LOOP_DETECTOR]: Added keyframe %d to keyframe_map, total keyframes: %zu\n",
              keyframe_id, keyframe_map_.size());

  // Add to BoW database
  bow_database_->addKeyframe(keyframe_id, descriptors);

  // Clean up old keyframes if buffer is full
  if (keyframe_history_.full()) {
    cleanupOldKeyframes();
  }

  // Update performance metrics
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    performance_metrics_.active_keyframes = static_cast<int>(keyframe_map_.size());
  }

  return true;
}

int LoopDetector::detectLoops(double current_timestamp, std::vector<LoopCandidate>& candidates) {
  candidates.clear();

  PRINT_DEBUG("[LOOP_DETECTOR]: detectLoops called with timestamp %.3f\n", current_timestamp);
  PRINT_DEBUG("[LOOP_DETECTOR]: Total keyframes in keyframe_map_: %zu\n", keyframe_map_.size());

  if (!is_initialized_ || is_shutdown_) {
    PRINT_DEBUG("[LOOP_DETECTOR]: detectLoops failed - not initialized (%s) or shutdown (%s)\n",
                is_initialized_ ? "true" : "false", is_shutdown_ ? "true" : "false");
    return 0;
  }

  // Find current keyframe
  const KeyframeData* current_keyframe = nullptr;
  for (const auto& kf_pair : keyframe_map_) {
    double time_diff = std::abs(kf_pair.second.timestamp - current_timestamp);
    PRINT_DEBUG("[LOOP_DETECTOR]: Checking keyframe %d: timestamp %.3f, time_diff %.3f\n",
                kf_pair.first, kf_pair.second.timestamp, time_diff);
    if (time_diff < 0.1) { // 100ms tolerance
      current_keyframe = &kf_pair.second;
      PRINT_DEBUG("[LOOP_DETECTOR]: Found matching keyframe %d\n", kf_pair.first);
      break;
    }
  }

  if (!current_keyframe) {
    PRINT_DEBUG("[LOOP_DETECTOR]: No matching keyframe found for timestamp %.3f\n", current_timestamp);
    return 0;
  }

  // Query BoW database for similar keyframes
  std::vector<LoopCandidate> bow_candidates;
  int num_bow_candidates = bow_database_->query(current_keyframe->descriptors,
                                               bow_candidates,
                                               options_.max_candidates_per_detection);

  if (num_bow_candidates == 0) {
    return 0;
  }

  // Filter candidates by temporal separation
  PRINT_DEBUG("[LOOP_DETECTOR]: Temporal filtering %zu BoW candidates with min_separation=%.1f\n",
              bow_candidates.size(), options_.min_loop_separation);
  std::vector<LoopCandidate> temporal_candidates;
  for (auto& candidate : bow_candidates) {
    const KeyframeData* match_keyframe = getKeyframe(candidate.match_keyframe_id);
    if (match_keyframe) {
      double time_diff = std::abs(current_timestamp - match_keyframe->timestamp);
      bool is_separated = isTemporallySeparated(current_timestamp, match_keyframe->timestamp);
      PRINT_DEBUG("[LOOP_DETECTOR]: Candidate %d: time_diff=%.1f, min_required=%.1f, passed=%s, similarity=%.3f\n",
                  candidate.match_keyframe_id, time_diff, options_.min_loop_separation,
                  is_separated ? "YES" : "NO", candidate.similarity_score);
      if (is_separated) {
        candidate.query_keyframe_id = next_keyframe_id_ - 1; // Most recent keyframe
        temporal_candidates.push_back(candidate);
      }
    }
  }
  PRINT_DEBUG("[LOOP_DETECTOR]: Temporal filtering result: %zu candidates passed\n", temporal_candidates.size());

  // Apply temporal consistency filtering
  applyTemporalConsistency(temporal_candidates);

  // Geometric verification for top candidates
  for (auto& candidate : temporal_candidates) {
    if (candidate.similarity_score > options_.bow_similarity_threshold) {
      Eigen::Matrix4d relative_pose;
      std::vector<cv::DMatch> inliers;
      double confidence;

      if (verifyLoopCandidate(candidate, relative_pose, inliers, confidence)) {
        candidate.relative_pose = relative_pose;
        candidate.inlier_matches = inliers;
        candidate.confidence = confidence;
        candidate.is_verified = true;
        candidates.push_back(candidate);
      }
    }
  }

  // Update performance metrics
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    performance_metrics_.total_loop_closures += static_cast<int>(candidates.size());

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_detection_time_);
    performance_metrics_.average_detection_time = duration.count();
    last_detection_time_ = now;
  }

  return static_cast<int>(candidates.size());
}

bool LoopDetector::verifyLoopCandidate(const LoopCandidate& candidate,
                                      Eigen::Matrix4d& relative_pose,
                                      std::vector<cv::DMatch>& inliers,
                                      double& confidence) {
  const KeyframeData* query_keyframe = getKeyframe(candidate.query_keyframe_id);
  const KeyframeData* match_keyframe = getKeyframe(candidate.match_keyframe_id);

  if (!query_keyframe || !match_keyframe) {
    return false;
  }

  return geometricVerification(*query_keyframe, *match_keyframe,
                              relative_pose, inliers, confidence);
}

bool LoopDetector::geometricVerification(const KeyframeData& keyframe1,
                                        const KeyframeData& keyframe2,
                                        Eigen::Matrix4d& relative_pose,
                                        std::vector<cv::DMatch>& inliers,
                                        double& confidence) {
  // Match descriptors between keyframes
  std::vector<cv::DMatch> matches;
  BriefExtractor::matchDescriptors(keyframe1.descriptors, keyframe2.descriptors, matches);

  if (matches.size() < static_cast<size_t>(options_.geometric_verification_threshold)) {
    return false;
  }

  // Convert keypoints to points for geometric verification
  std::vector<cv::Point2f> points1, points2;
  for (const auto& match : matches) {
    points1.push_back(keyframe1.keypoints[match.queryIdx].pt);
    points2.push_back(keyframe2.keypoints[match.trainIdx].pt);
  }

  // Estimate essential matrix
  cv::Mat essential_matrix;
  std::vector<int> inlier_indices;
  if (!estimateEssentialMatrix(points1, points2, essential_matrix, inlier_indices)) {
    return false;
  }

  // Check if enough inliers
  if (inlier_indices.size() < static_cast<size_t>(options_.geometric_verification_threshold)) {
    return false;
  }

  // Recover pose from essential matrix
  cv::Mat R, t;
  if (!recoverPose(essential_matrix, points1, points2, inlier_indices, R, t)) {
    return false;
  }

  // Convert to Eigen format
  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R_eigen(i, j) = R.at<double>(i, j);
    }
    t_eigen(i) = t.at<double>(i);
  }

  // Construct relative pose
  relative_pose = Eigen::Matrix4d::Identity();
  relative_pose.block<3, 3>(0, 0) = R_eigen;
  relative_pose.block<3, 1>(0, 3) = t_eigen;

  // Convert inlier indices to DMatch format
  inliers.clear();
  for (int idx : inlier_indices) {
    inliers.push_back(matches[idx]);
  }

  // Compute confidence based on inlier ratio
  confidence = static_cast<double>(inlier_indices.size()) / static_cast<double>(matches.size());

  return confidence > 0.3; // At least 30% inliers
}

bool LoopDetector::estimateEssentialMatrix(const std::vector<cv::Point2f>& points1,
                                          const std::vector<cv::Point2f>& points2,
                                          cv::Mat& essential_matrix,
                                          std::vector<int>& inliers) {
  if (points1.size() != points2.size() || points1.size() < 8) {
    return false;
  }

  // Camera intrinsic parameters (assuming calibrated camera with unit focal length)
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat inlier_mask;
  essential_matrix = cv::findEssentialMat(points1, points2, camera_matrix,
                                         cv::RANSAC, options_.ransac_probability,
                                         options_.essential_matrix_threshold, inlier_mask);

  if (essential_matrix.empty()) {
    return false;
  }

  // Extract inlier indices
  inliers.clear();
  for (int i = 0; i < inlier_mask.rows; ++i) {
    if (inlier_mask.at<uchar>(i)) {
      inliers.push_back(i);
    }
  }

  return inliers.size() >= 8;
}

bool LoopDetector::recoverPose(const cv::Mat& essential_matrix,
                              const std::vector<cv::Point2f>& points1,
                              const std::vector<cv::Point2f>& points2,
                              const std::vector<int>& inliers,
                              cv::Mat& R, cv::Mat& t) {
  if (inliers.size() < 8) {
    return false;
  }

  // Extract inlier points
  std::vector<cv::Point2f> inlier_points1, inlier_points2;
  for (int idx : inliers) {
    inlier_points1.push_back(points1[idx]);
    inlier_points2.push_back(points2[idx]);
  }

  // Camera intrinsic parameters
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

  cv::Mat inlier_mask;
  int num_inliers = cv::recoverPose(essential_matrix, inlier_points1, inlier_points2,
                                   camera_matrix, R, t, inlier_mask);

  return num_inliers > 0;
}

void LoopDetector::applyTemporalConsistency(std::vector<LoopCandidate>& candidates) {
  std::vector<LoopCandidate> consistent_candidates;

  for (auto& candidate : candidates) {
    std::pair<int, int> key(candidate.query_keyframe_id, candidate.match_keyframe_id);

    // Update consistency count
    consistency_map_[key]++;

    // Check if candidate meets consistency requirement
    if (consistency_map_[key] >= options_.temporal_consistency_window) {
      candidate.temporal_consistency_score = static_cast<double>(consistency_map_[key]) /
                                           options_.temporal_consistency_window;
      consistent_candidates.push_back(candidate);
    }
  }

  // Clean up old consistency data
  const int max_consistency_entries = 10000;
  if (consistency_map_.size() > max_consistency_entries) {
    // Remove oldest entries (simplified cleanup)
    auto it = consistency_map_.begin();
    std::advance(it, max_consistency_entries / 2);
    consistency_map_.erase(consistency_map_.begin(), it);
  }

  candidates = consistent_candidates;
}

void LoopDetector::updatePerformanceMetrics(double cpu_usage, double memory_usage) {
  {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    performance_metrics_.cpu_usage = cpu_usage;
    performance_metrics_.memory_usage = memory_usage;
  }

  if (options_.enable_resource_monitoring) {
    adaptParameters();
  }
}

PerformanceMetrics LoopDetector::getPerformanceMetrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return performance_metrics_;
}

void LoopDetector::detectionThreadFunc() {
  PRINT_INFO("[LOOP]: Detection thread started\n");

  while (!is_shutdown_) {
    DetectionTask task(KeyframeData(), 0.0, -1);
    bool has_task = false;

    // Wait for task or shutdown signal
    {
      std::unique_lock<std::mutex> lock(detection_mutex_);
      detection_cv_.wait(lock, [this] { return !detection_queue_.empty() || is_shutdown_; });

      if (!detection_queue_.empty()) {
        task = detection_queue_.front();
        detection_queue_.pop();
        has_task = true;
      }
    }

    if (has_task) {
      auto candidates = processDetectionTask(task);
      // Results would typically be stored or signaled back to main thread
      // For now, we just process the task
    }
  }

  PRINT_INFO("[LOOP]: Detection thread stopped\n");
}

std::vector<LoopCandidate> LoopDetector::processDetectionTask(const DetectionTask& task) {
  std::vector<LoopCandidate> candidates;

  // Query BoW database
  bow_database_->query(task.keyframe.descriptors, candidates,
                      options_.max_candidates_per_detection);

  // Apply temporal filtering
  std::vector<LoopCandidate> filtered_candidates;
  for (const auto& candidate : candidates) {
    const KeyframeData* match_keyframe = getKeyframe(candidate.match_keyframe_id);
    if (match_keyframe && isTemporallySeparated(task.current_timestamp, match_keyframe->timestamp)) {
      filtered_candidates.push_back(candidate);
    }
  }

  return filtered_candidates;
}

void LoopDetector::adaptParameters() {
  std::lock_guard<std::mutex> lock(metrics_mutex_);

  if (performance_metrics_.cpu_usage > options_.cpu_usage_threshold) {
    // Reduce detection frequency to save CPU
    if (options_.adaptive_frequency_scaling) {
      options_.detection_frequency = std::max(0.1, options_.detection_frequency * 0.8);
    }
  }

  if (performance_metrics_.memory_usage > options_.memory_usage_threshold) {
    // Reduce max keyframes to save memory
    options_.max_keyframes_memory = std::max(100, options_.max_keyframes_memory - 50);
    cleanupOldKeyframes();
  }
}

void LoopDetector::cleanupOldKeyframes() {
  const int target_size = static_cast<int>(options_.max_keyframes_memory * 0.8);

  if (static_cast<int>(keyframe_map_.size()) > target_size) {
    // Remove oldest keyframes
    std::vector<int> keyframes_to_remove;
    auto it = keyframe_map_.begin();
    int to_remove = static_cast<int>(keyframe_map_.size()) - target_size;

    for (int i = 0; i < to_remove && it != keyframe_map_.end(); ++i, ++it) {
      keyframes_to_remove.push_back(it->first);
    }

    // Remove from map and database
    for (int keyframe_id : keyframes_to_remove) {
      keyframe_map_.erase(keyframe_id);
    }
    bow_database_->removeKeyframes(keyframes_to_remove);
  }
}

bool LoopDetector::isTemporallySeparated(double t1, double t2) const {
  return std::abs(t1 - t2) > options_.min_loop_separation;
}

void LoopDetector::keypointsToPoints(const std::vector<cv::KeyPoint>& keypoints,
                                    std::vector<cv::Point2f>& points) {
  points.clear();
  points.reserve(keypoints.size());
  for (const auto& kp : keypoints) {
    points.push_back(kp.pt);
  }
}

const KeyframeData* LoopDetector::getKeyframe(int keyframe_id) const {
  auto it = keyframe_map_.find(keyframe_id);
  return (it != keyframe_map_.end()) ? &it->second : nullptr;
}