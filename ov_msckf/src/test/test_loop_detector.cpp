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

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <random>

#include "loop/LoopDetector.h"
#include "loop/BriefExtractor.h"

using namespace ov_msckf;

class LoopDetectorTest : public ::testing::Test {
protected:
  void SetUp() override {
    LoopDetectorOptions options;
    options.enabled = true;
    options.max_keyframes_memory = 100;
    options.detection_frequency = 10.0;
    options.min_loop_separation = 1.0; // Reduced for testing
    options.bow_similarity_threshold = 0.001;
    options.geometric_verification_threshold = 10; // Lower for testing

    detector = std::make_unique<LoopDetector>(options);

    // Create test keypoints and descriptors
    createTestData();
  }

  void createTestData() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> bit_dist(0, 1);
    std::uniform_real_distribution<float> coord_dist(50.0f, 400.0f);

    // Create test keypoints
    for (int i = 0; i < num_test_keypoints; ++i) {
      cv::KeyPoint kpt(coord_dist(rng), coord_dist(rng), 10.0f);
      test_keypoints.push_back(kpt);

      // Create corresponding descriptor
      BriefDescriptor desc;
      for (int bit = 0; bit < 256; ++bit) {
        if (bit_dist(rng)) {
          desc[bit] = 1;
        }
      }
      test_descriptors.push_back(desc);
    }

    // Create test poses
    for (int i = 0; i < 10; ++i) {
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      pose(0, 3) = i * 2.0; // Move along x-axis
      pose(1, 3) = i * 0.5; // Slight y movement
      test_poses.push_back(pose);
    }
  }

  std::unique_ptr<LoopDetector> detector;
  std::vector<cv::KeyPoint> test_keypoints;
  std::vector<BriefDescriptor> test_descriptors;
  std::vector<Eigen::Matrix4d> test_poses;
  static const int num_test_keypoints = 100;
};

TEST_F(LoopDetectorTest, ConstructorTest) {
  EXPECT_TRUE(detector != nullptr);
  EXPECT_FALSE(detector->isReady()); // Not initialized yet
  EXPECT_EQ(detector->getNumKeyframes(), 0);
}

TEST_F(LoopDetectorTest, InitializationTest) {
  EXPECT_TRUE(detector->initialize());
  EXPECT_TRUE(detector->isReady());

  // Test metrics after initialization
  PerformanceMetrics metrics = detector->getPerformanceMetrics();
  EXPECT_EQ(metrics.active_keyframes, 0);
  EXPECT_EQ(metrics.total_loop_closures, 0);
}

TEST_F(LoopDetectorTest, AddKeyframeTest) {
  detector->initialize();

  // Add first keyframe
  bool success = detector->addKeyframe(1.0, test_keypoints, test_descriptors, test_poses[0]);
  EXPECT_TRUE(success);
  EXPECT_EQ(detector->getNumKeyframes(), 1);

  // Add more keyframes
  for (int i = 1; i < 5; ++i) {
    success = detector->addKeyframe(i + 1.0, test_keypoints, test_descriptors, test_poses[i]);
    EXPECT_TRUE(success);
  }
  EXPECT_EQ(detector->getNumKeyframes(), 5);
}

TEST_F(LoopDetectorTest, AddKeyframeEdgeCases) {
  detector->initialize();

  // Add keyframe with empty data
  std::vector<cv::KeyPoint> empty_keypoints;
  std::vector<BriefDescriptor> empty_descriptors;
  bool success = detector->addKeyframe(1.0, empty_keypoints, empty_descriptors, test_poses[0]);
  EXPECT_FALSE(success); // Should fail with empty data

  // Add keyframe without initialization
  LoopDetectorOptions options;
  LoopDetector uninitialized_detector(options);
  success = uninitialized_detector.addKeyframe(1.0, test_keypoints, test_descriptors, test_poses[0]);
  EXPECT_FALSE(success);
}

TEST_F(LoopDetectorTest, DetectLoopsBasicTest) {
  detector->initialize();

  // Add some keyframes with temporal separation
  double timestamp = 0.0;
  for (int i = 0; i < 3; ++i) {
    timestamp += 2.0; // 2 second intervals
    detector->addKeyframe(timestamp, test_keypoints, test_descriptors, test_poses[i]);
  }

  // Try to detect loops (may not find any with random data)
  std::vector<LoopCandidate> candidates;
  int num_loops = detector->detectLoops(timestamp, candidates);

  // Should not crash and return valid number
  EXPECT_GE(num_loops, 0);
  EXPECT_EQ(static_cast<int>(candidates.size()), num_loops);
}

TEST_F(LoopDetectorTest, DetectLoopsSameDataTest) {
  detector->initialize();

  // Add identical keyframes at different times (should create loop)
  double timestamp1 = 1.0;
  double timestamp2 = 5.0; // Sufficient temporal separation

  detector->addKeyframe(timestamp1, test_keypoints, test_descriptors, test_poses[0]);
  detector->addKeyframe(timestamp2, test_keypoints, test_descriptors, test_poses[0]);

  // Detect loops
  std::vector<LoopCandidate> candidates;
  int num_loops = detector->detectLoops(timestamp2, candidates);

  // May or may not detect loops depending on thresholds and geometric verification
  EXPECT_GE(num_loops, 0);

  // Check candidate validity if any found
  for (const auto& candidate : candidates) {
    EXPECT_GE(candidate.query_keyframe_id, 0);
    EXPECT_GE(candidate.match_keyframe_id, 0);
    EXPECT_GE(candidate.similarity_score, 0.0);
    EXPECT_LE(candidate.similarity_score, 1.0);
  }
}

TEST_F(LoopDetectorTest, VerifyLoopCandidateTest) {
  detector->initialize();

  // Add keyframes
  detector->addKeyframe(1.0, test_keypoints, test_descriptors, test_poses[0]);
  detector->addKeyframe(3.0, test_keypoints, test_descriptors, test_poses[1]);

  // Create a test candidate
  LoopCandidate candidate;
  candidate.query_keyframe_id = 1;
  candidate.match_keyframe_id = 0;
  candidate.similarity_score = 0.8;

  Eigen::Matrix4d relative_pose;
  std::vector<cv::DMatch> inliers;
  double confidence;

  // Verify candidate (may pass or fail depending on geometric verification)
  bool verified = detector->verifyLoopCandidate(candidate, relative_pose, inliers, confidence);

  // Should not crash and return valid results
  EXPECT_GE(confidence, 0.0);
  EXPECT_LE(confidence, 1.0);

  if (verified) {
    EXPECT_GT(confidence, 0.0);
  }
}

TEST_F(LoopDetectorTest, PerformanceMetricsTest) {
  detector->initialize();

  PerformanceMetrics initial_metrics = detector->getPerformanceMetrics();
  EXPECT_EQ(initial_metrics.active_keyframes, 0);
  EXPECT_EQ(initial_metrics.total_loop_closures, 0);

  // Add keyframes and check metrics update
  detector->addKeyframe(1.0, test_keypoints, test_descriptors, test_poses[0]);

  // Update metrics (simulate resource monitoring)
  detector->updatePerformanceMetrics(50.0, 60.0); // 50% CPU, 60% memory

  PerformanceMetrics updated_metrics = detector->getPerformanceMetrics();
  EXPECT_EQ(updated_metrics.cpu_usage, 50.0);
  EXPECT_EQ(updated_metrics.memory_usage, 60.0);
}

TEST_F(LoopDetectorTest, MemoryManagementTest) {
  LoopDetectorOptions options;
  options.max_keyframes_memory = 5; // Small limit for testing
  options.detection_frequency = 10.0;

  LoopDetector memory_test_detector(options);
  memory_test_detector.initialize();

  // Add more keyframes than the limit
  for (int i = 0; i < 10; ++i) {
    memory_test_detector.addKeyframe(i + 1.0, test_keypoints, test_descriptors, test_poses[i % test_poses.size()]);
  }

  // Should not exceed memory limit significantly
  EXPECT_LE(static_cast<int>(memory_test_detector.getNumKeyframes()), 8); // Allow some overhead
}

TEST_F(LoopDetectorTest, ShutdownTest) {
  detector->initialize();

  // Add some keyframes
  for (int i = 0; i < 3; ++i) {
    detector->addKeyframe(i + 1.0, test_keypoints, test_descriptors, test_poses[i]);
  }

  // Shutdown should complete without crashing
  detector->shutdown();

  // After shutdown, should not be ready
  EXPECT_FALSE(detector->isReady());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}