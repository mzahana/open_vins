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

#ifndef OV_MSCKF_BRIEF_EXTRACTOR_H
#define OV_MSCKF_BRIEF_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "LoopTypes.h"

namespace ov_msckf {

/**
 * @brief Efficient BRIEF descriptor extractor optimized for loop closure
 *
 * This class implements a lightweight BRIEF descriptor extractor that is 3x faster
 * than ORB descriptors while maintaining good discriminative properties for loop
 * closure detection. The implementation uses pre-computed sampling patterns and
 * efficient bit manipulation for real-time performance.
 */
class BriefExtractor {

public:
  /**
   * @brief Constructor with configurable parameters
   * @param patch_size Size of the patch around each keypoint (default: 48)
   * @param descriptor_length Length of the binary descriptor in bits (default: 256)
   */
  explicit BriefExtractor(int patch_size = 48, int descriptor_length = 256);

  /**
   * @brief Extract BRIEF descriptors from keypoints
   * @param image Input grayscale image
   * @param keypoints Vector of keypoints to describe
   * @param descriptors Output vector of BRIEF descriptors
   */
  void extract(const cv::Mat& image,
               const std::vector<cv::KeyPoint>& keypoints,
               std::vector<BriefDescriptor>& descriptors);

  /**
   * @brief Compute Hamming distance between two BRIEF descriptors
   * @param desc1 First descriptor
   * @param desc2 Second descriptor
   * @return Hamming distance (0-256)
   */
  static int hammingDistance(const BriefDescriptor& desc1, const BriefDescriptor& desc2);

  /**
   * @brief Match descriptors using brute-force with cross-check
   * @param descriptors1 First set of descriptors
   * @param descriptors2 Second set of descriptors
   * @param matches Output matches
   * @param max_distance Maximum Hamming distance for valid matches
   */
  static void matchDescriptors(const std::vector<BriefDescriptor>& descriptors1,
                               const std::vector<BriefDescriptor>& descriptors2,
                               std::vector<cv::DMatch>& matches,
                               int max_distance = 64);

private:
  int patch_size_;
  int descriptor_length_;

  // Pre-computed sampling pattern for efficiency
  std::vector<cv::Point2f> sampling_pattern_;

  /**
   * @brief Generate random sampling pattern for BRIEF
   */
  void generateSamplingPattern();

  /**
   * @brief Smooth image patch to reduce noise sensitivity
   * @param image Input image
   * @param smoothed Output smoothed image
   */
  void smoothImage(const cv::Mat& image, cv::Mat& smoothed);

  /**
   * @brief Extract single descriptor from keypoint
   * @param smoothed_image Smoothed input image
   * @param keypoint Keypoint to describe
   * @param descriptor Output descriptor
   * @return true if successful, false if keypoint too close to border
   */
  bool extractSingleDescriptor(const cv::Mat& smoothed_image,
                               const cv::KeyPoint& keypoint,
                               BriefDescriptor& descriptor);
};

} // namespace ov_msckf

#endif // OV_MSCKF_BRIEF_EXTRACTOR_H