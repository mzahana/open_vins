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
#include <vector>

#include "loop/BriefExtractor.h"

using namespace ov_msckf;

class BriefExtractorTest : public ::testing::Test {
protected:
  void SetUp() override {
    extractor = std::make_unique<BriefExtractor>();

    // Create a test image with some pattern
    test_image = cv::Mat::zeros(480, 640, CV_8UC1);

    // Add some patterns to make it interesting
    cv::rectangle(test_image, cv::Rect(100, 100, 50, 50), cv::Scalar(255), -1);
    cv::rectangle(test_image, cv::Rect(200, 200, 30, 30), cv::Scalar(128), -1);
    cv::circle(test_image, cv::Point(400, 300), 25, cv::Scalar(200), -1);

    // Add some noise
    cv::Mat noise(test_image.size(), CV_8UC1);
    cv::randu(noise, 0, 50);
    test_image += noise;

    // Extract FAST keypoints
    cv::FAST(test_image, test_keypoints, 30, true);
  }

  std::unique_ptr<BriefExtractor> extractor;
  cv::Mat test_image;
  std::vector<cv::KeyPoint> test_keypoints;
};

TEST_F(BriefExtractorTest, ConstructorTest) {
  EXPECT_TRUE(extractor != nullptr);

  // Test with custom parameters
  BriefExtractor custom_extractor(64, 512);
  EXPECT_TRUE(true); // Constructor should not throw
}

TEST_F(BriefExtractorTest, ExtractDescriptorsBasic) {
  std::vector<BriefDescriptor> descriptors;

  // Should not crash with empty inputs
  extractor->extract(cv::Mat(), std::vector<cv::KeyPoint>(), descriptors);
  EXPECT_TRUE(descriptors.empty());

  // Extract from test image
  extractor->extract(test_image, test_keypoints, descriptors);

  // Should have extracted descriptors
  EXPECT_EQ(descriptors.size(), test_keypoints.size());

  // Check that descriptors are not all zeros
  bool found_non_zero = false;
  for (const auto& desc : descriptors) {
    if (desc.count() > 0) {
      found_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(found_non_zero);
}

TEST_F(BriefExtractorTest, ExtractDescriptorsEdgeCases) {
  std::vector<BriefDescriptor> descriptors;

  // Test with keypoints near image borders
  std::vector<cv::KeyPoint> edge_keypoints;
  edge_keypoints.emplace_back(5, 5, 10);     // Near top-left corner
  edge_keypoints.emplace_back(635, 5, 10);   // Near top-right corner
  edge_keypoints.emplace_back(5, 475, 10);   // Near bottom-left corner
  edge_keypoints.emplace_back(635, 475, 10); // Near bottom-right corner

  extractor->extract(test_image, edge_keypoints, descriptors);

  // Some descriptors might be empty due to border effects
  EXPECT_EQ(descriptors.size(), edge_keypoints.size());
}

TEST_F(BriefExtractorTest, HammingDistanceTest) {
  BriefDescriptor desc1, desc2, desc3;

  // Set some bits
  desc1[0] = desc1[10] = desc1[50] = 1;
  desc2[0] = desc2[20] = desc2[60] = 1;
  desc3 = desc1; // Copy

  // Distance between identical descriptors should be 0
  EXPECT_EQ(BriefExtractor::hammingDistance(desc1, desc3), 0);

  // Distance should be symmetric
  int dist12 = BriefExtractor::hammingDistance(desc1, desc2);
  int dist21 = BriefExtractor::hammingDistance(desc2, desc1);
  EXPECT_EQ(dist12, dist21);

  // Distance should be reasonable
  EXPECT_GT(dist12, 0);
  EXPECT_LT(dist12, 256); // Should be less than descriptor length
}

TEST_F(BriefExtractorTest, MatchDescriptorsTest) {
  // Create some test descriptors
  std::vector<BriefDescriptor> descriptors1, descriptors2;

  // Extract descriptors from test image
  extractor->extract(test_image, test_keypoints, descriptors1);

  // Create slightly modified image for second set
  cv::Mat test_image2 = test_image.clone();
  cv::GaussianBlur(test_image2, test_image2, cv::Size(3, 3), 1.0);

  std::vector<cv::KeyPoint> keypoints2 = test_keypoints; // Same keypoints
  extractor->extract(test_image2, keypoints2, descriptors2);

  // Match descriptors
  std::vector<cv::DMatch> matches;
  BriefExtractor::matchDescriptors(descriptors1, descriptors2, matches);

  // Should have some matches
  EXPECT_GT(matches.size(), 0);

  // Check match validity
  for (const auto& match : matches) {
    EXPECT_GE(match.queryIdx, 0);
    EXPECT_LT(match.queryIdx, static_cast<int>(descriptors1.size()));
    EXPECT_GE(match.trainIdx, 0);
    EXPECT_LT(match.trainIdx, static_cast<int>(descriptors2.size()));
    EXPECT_GE(match.distance, 0);
  }
}

TEST_F(BriefExtractorTest, MatchDescriptorsEmptyTest) {
  std::vector<BriefDescriptor> empty_descriptors1, empty_descriptors2;
  std::vector<cv::DMatch> matches;

  // Should handle empty input gracefully
  BriefExtractor::matchDescriptors(empty_descriptors1, empty_descriptors2, matches);
  EXPECT_TRUE(matches.empty());
}

TEST_F(BriefExtractorTest, RepeatabilityTest) {
  std::vector<BriefDescriptor> descriptors1, descriptors2;

  // Extract descriptors twice from same image and keypoints
  extractor->extract(test_image, test_keypoints, descriptors1);
  extractor->extract(test_image, test_keypoints, descriptors2);

  // Results should be identical
  EXPECT_EQ(descriptors1.size(), descriptors2.size());

  for (size_t i = 0; i < descriptors1.size(); ++i) {
    EXPECT_EQ(descriptors1[i], descriptors2[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}