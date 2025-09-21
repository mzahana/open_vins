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

#ifndef OV_MSCKF_LOOP_DETECTOR_H
#define OV_MSCKF_LOOP_DETECTOR_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>

#include "BowDatabase.h"
#include "BriefExtractor.h"
#include "LoopTypes.h"

namespace ov_msckf {

/**
 * @brief Main loop detection class with asynchronous processing
 *
 * This class provides efficient loop closure detection using BRIEF descriptors
 * and a bag-of-words database. It features:
 * - Asynchronous keyframe processing for real-time performance
 * - Geometric verification with essential matrix estimation
 * - Temporal consistency filtering to reduce false positives
 * - Resource monitoring and adaptive parameter adjustment
 * - Multi-hypothesis loop closure support
 */
class LoopDetector {

public:
  /**
   * @brief Constructor with options
   * @param options Loop detector configuration options
   */
  explicit LoopDetector(const LoopDetectorOptions& options);

  /**
   * @brief Destructor
   */
  ~LoopDetector();

  /**
   * @brief Initialize loop detector (load vocabulary, start threads)
   * @return true if successful, false otherwise
   */
  bool initialize();

  /**
   * @brief Shutdown loop detector (stop threads, save state)
   */
  void shutdown();

  /**
   * @brief Add keyframe for loop detection
   * @param timestamp Keyframe timestamp
   * @param keypoints Detected keypoints
   * @param descriptors BRIEF descriptors
   * @param pose Camera pose in world frame
   * @return true if keyframe was added successfully
   */
  bool addKeyframe(double timestamp,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const std::vector<BriefDescriptor>& descriptors,
                   const Eigen::Matrix4d& pose);

  /**
   * @brief Detect loop closures for current keyframe
   * @param current_timestamp Current timestamp
   * @param candidates Output vector of loop candidates
   * @return Number of loop candidates found
   */
  int detectLoops(double current_timestamp, std::vector<LoopCandidate>& candidates);

  /**
   * @brief Verify loop closure candidate with geometric constraints
   * @param candidate Loop candidate to verify
   * @param relative_pose Output relative pose between keyframes
   * @param inliers Output inlier matches
   * @param confidence Output confidence score
   * @return true if loop closure is verified
   */
  bool verifyLoopCandidate(const LoopCandidate& candidate,
                          Eigen::Matrix4d& relative_pose,
                          std::vector<cv::DMatch>& inliers,
                          double& confidence);

  /**
   * @brief Update performance metrics and adapt parameters if needed
   * @param cpu_usage Current CPU usage percentage
   * @param memory_usage Current memory usage percentage
   */
  void updatePerformanceMetrics(double cpu_usage, double memory_usage);

  /**
   * @brief Get current performance metrics
   * @return Performance metrics structure
   */
  PerformanceMetrics getPerformanceMetrics() const;

  /**
   * @brief Check if detector is initialized and ready
   * @return true if ready, false otherwise
   */
  bool isReady() const { return is_initialized_ && !is_shutdown_ && bow_database_->isInitialized(); }

  /**
   * @brief Get number of stored keyframes
   * @return Number of keyframes in memory
   */
  size_t getNumKeyframes() const { return keyframe_history_.size(); }

private:
  LoopDetectorOptions options_;
  std::atomic<bool> is_initialized_;
  std::atomic<bool> is_shutdown_;

  // Core components
  std::unique_ptr<BowDatabase> bow_database_;
  std::unique_ptr<BriefExtractor> brief_extractor_;

  // Keyframe storage
  CircularBuffer<KeyframeData> keyframe_history_;
  std::map<int, KeyframeData> keyframe_map_; // keyframe_id -> KeyframeData
  int next_keyframe_id_;

  // Threading components
  std::thread detection_thread_;
  std::queue<DetectionTask> detection_queue_;
  std::mutex detection_mutex_;
  std::condition_variable detection_cv_;
  std::atomic<int> next_task_id_;

  // Temporal consistency tracking
  std::map<std::pair<int, int>, int> consistency_map_; // (query_id, match_id) -> count
  std::map<int, std::vector<int>> recent_detections_; // query_id -> match_ids

  // Performance monitoring
  mutable std::mutex metrics_mutex_;
  PerformanceMetrics performance_metrics_;
  std::chrono::steady_clock::time_point last_detection_time_;

  /**
   * @brief Main detection thread function
   */
  void detectionThreadFunc();

  /**
   * @brief Process single detection task
   * @param task Detection task to process
   * @return Vector of detected loop candidates
   */
  std::vector<LoopCandidate> processDetectionTask(const DetectionTask& task);

  /**
   * @brief Apply temporal consistency filtering
   * @param candidates Input/output candidates
   */
  void applyTemporalConsistency(std::vector<LoopCandidate>& candidates);

  /**
   * @brief Extract features from image using FAST detector
   * @param image Input image
   * @param keypoints Output keypoints
   * @param descriptors Output BRIEF descriptors
   */
  void extractFeatures(const cv::Mat& image,
                      std::vector<cv::KeyPoint>& keypoints,
                      std::vector<BriefDescriptor>& descriptors);

  /**
   * @brief Geometric verification using essential matrix
   * @param keyframe1 First keyframe
   * @param keyframe2 Second keyframe
   * @param relative_pose Output relative pose
   * @param inliers Output inlier matches
   * @param confidence Output confidence score
   * @return true if verification successful
   */
  bool geometricVerification(const KeyframeData& keyframe1,
                            const KeyframeData& keyframe2,
                            Eigen::Matrix4d& relative_pose,
                            std::vector<cv::DMatch>& inliers,
                            double& confidence);

  /**
   * @brief Estimate essential matrix with RANSAC
   * @param points1 Points from first image
   * @param points2 Points from second image
   * @param essential_matrix Output essential matrix
   * @param inliers Output inlier indices
   * @return true if successful
   */
  bool estimateEssentialMatrix(const std::vector<cv::Point2f>& points1,
                              const std::vector<cv::Point2f>& points2,
                              cv::Mat& essential_matrix,
                              std::vector<int>& inliers);

  /**
   * @brief Recover pose from essential matrix
   * @param essential_matrix Input essential matrix
   * @param points1 Points from first image
   * @param points2 Points from second image
   * @param inliers Inlier indices
   * @param R Output rotation matrix
   * @param t Output translation vector
   * @return true if successful
   */
  bool recoverPose(const cv::Mat& essential_matrix,
                  const std::vector<cv::Point2f>& points1,
                  const std::vector<cv::Point2f>& points2,
                  const std::vector<int>& inliers,
                  cv::Mat& R, cv::Mat& t);

  /**
   * @brief Adapt parameters based on system performance
   */
  void adaptParameters();

  /**
   * @brief Clean up old keyframes to manage memory
   */
  void cleanupOldKeyframes();

  /**
   * @brief Check if two timestamps are temporally separated enough
   * @param t1 First timestamp
   * @param t2 Second timestamp
   * @return true if separation is sufficient
   */
  bool isTemporallySeparated(double t1, double t2) const;

  /**
   * @brief Convert keypoints to points for geometric verification
   * @param keypoints Input keypoints
   * @param points Output 2D points
   */
  static void keypointsToPoints(const std::vector<cv::KeyPoint>& keypoints,
                               std::vector<cv::Point2f>& points);

  /**
   * @brief Get keyframe by ID
   * @param keyframe_id Keyframe identifier
   * @return Pointer to keyframe data or nullptr if not found
   */
  const KeyframeData* getKeyframe(int keyframe_id) const;
};

} // namespace ov_msckf

#endif // OV_MSCKF_LOOP_DETECTOR_H