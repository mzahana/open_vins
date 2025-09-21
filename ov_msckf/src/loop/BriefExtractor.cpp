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

#include "BriefExtractor.h"
#include <random>

using namespace ov_msckf;

BriefExtractor::BriefExtractor(int patch_size, int descriptor_length)
  : patch_size_(patch_size), descriptor_length_(descriptor_length) {
  generateSamplingPattern();
}

void BriefExtractor::generateSamplingPattern() {
  sampling_pattern_.clear();
  sampling_pattern_.reserve(descriptor_length_ * 2);

  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::normal_distribution<float> dist(0.0f, patch_size_ / 5.0f);

  for (int i = 0; i < descriptor_length_; ++i) {
    cv::Point2f p1(dist(rng), dist(rng));
    cv::Point2f p2(dist(rng), dist(rng));

    sampling_pattern_.push_back(p1);
    sampling_pattern_.push_back(p2);
  }
}

void BriefExtractor::extract(const cv::Mat& image,
                            const std::vector<cv::KeyPoint>& keypoints,
                            std::vector<BriefDescriptor>& descriptors) {
  if (image.empty() || keypoints.empty()) {
    descriptors.clear();
    return;
  }

  cv::Mat smoothed_image;
  smoothImage(image, smoothed_image);

  descriptors.clear();
  descriptors.reserve(keypoints.size());

  for (const auto& keypoint : keypoints) {
    BriefDescriptor descriptor;
    if (extractSingleDescriptor(smoothed_image, keypoint, descriptor)) {
      descriptors.push_back(descriptor);
    } else {
      descriptors.push_back(BriefDescriptor()); // Empty descriptor for invalid keypoints
    }
  }
}

void BriefExtractor::smoothImage(const cv::Mat& image, cv::Mat& smoothed) {
  cv::GaussianBlur(image, smoothed, cv::Size(9, 9), 2, 2, cv::BORDER_REFLECT_101);
}

bool BriefExtractor::extractSingleDescriptor(const cv::Mat& smoothed_image,
                                            const cv::KeyPoint& keypoint,
                                            BriefDescriptor& descriptor) {
  const float x = keypoint.pt.x;
  const float y = keypoint.pt.y;
  const int half_patch = patch_size_ / 2;

  // Check if keypoint is too close to image borders
  if (x < half_patch || y < half_patch ||
      x >= smoothed_image.cols - half_patch || y >= smoothed_image.rows - half_patch) {
    return false;
  }

  descriptor.reset();

  // Extract descriptor using sampling pattern
  for (int i = 0; i < descriptor_length_; ++i) {
    const cv::Point2f& p1 = sampling_pattern_[2 * i];
    const cv::Point2f& p2 = sampling_pattern_[2 * i + 1];

    int x1 = static_cast<int>(x + p1.x + 0.5f);
    int y1 = static_cast<int>(y + p1.y + 0.5f);
    int x2 = static_cast<int>(x + p2.x + 0.5f);
    int y2 = static_cast<int>(y + p2.y + 0.5f);

    // Bounds check
    x1 = std::max(0, std::min(smoothed_image.cols - 1, x1));
    y1 = std::max(0, std::min(smoothed_image.rows - 1, y1));
    x2 = std::max(0, std::min(smoothed_image.cols - 1, x2));
    y2 = std::max(0, std::min(smoothed_image.rows - 1, y2));

    if (smoothed_image.at<uchar>(y1, x1) < smoothed_image.at<uchar>(y2, x2)) {
      descriptor[i] = 1;
    }
  }

  return true;
}

int BriefExtractor::hammingDistance(const BriefDescriptor& desc1, const BriefDescriptor& desc2) {
  return static_cast<int>((desc1 ^ desc2).count());
}

void BriefExtractor::matchDescriptors(const std::vector<BriefDescriptor>& descriptors1,
                                     const std::vector<BriefDescriptor>& descriptors2,
                                     std::vector<cv::DMatch>& matches,
                                     int max_distance) {
  matches.clear();

  if (descriptors1.empty() || descriptors2.empty()) {
    return;
  }

  // Brute-force matching with cross-check
  for (size_t i = 0; i < descriptors1.size(); ++i) {
    int best_distance = std::numeric_limits<int>::max();
    int second_best_distance = std::numeric_limits<int>::max();
    int best_idx = -1;

    // Find best and second-best matches
    for (size_t j = 0; j < descriptors2.size(); ++j) {
      int distance = hammingDistance(descriptors1[i], descriptors2[j]);

      if (distance < best_distance) {
        second_best_distance = best_distance;
        best_distance = distance;
        best_idx = static_cast<int>(j);
      } else if (distance < second_best_distance) {
        second_best_distance = distance;
      }
    }

    // Apply distance threshold and ratio test
    if (best_distance <= max_distance && best_idx >= 0) {
      // Ratio test to reduce ambiguous matches
      float ratio = static_cast<float>(best_distance) / static_cast<float>(second_best_distance);
      if (ratio < 0.8f) {
        // Cross-check: verify that best match in descriptors2 also matches back to descriptors1
        int cross_check_distance = std::numeric_limits<int>::max();
        int cross_check_idx = -1;

        for (size_t k = 0; k < descriptors1.size(); ++k) {
          int distance = hammingDistance(descriptors2[best_idx], descriptors1[k]);
          if (distance < cross_check_distance) {
            cross_check_distance = distance;
            cross_check_idx = static_cast<int>(k);
          }
        }

        if (cross_check_idx == static_cast<int>(i)) {
          matches.emplace_back(static_cast<int>(i), best_idx, static_cast<float>(best_distance));
        }
      }
    }
  }
}