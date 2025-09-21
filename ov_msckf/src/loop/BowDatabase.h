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

#ifndef OV_MSCKF_BOW_DATABASE_H
#define OV_MSCKF_BOW_DATABASE_H

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "LoopTypes.h"

namespace ov_msckf {

/**
 * @brief Bag-of-Words database for efficient place recognition
 *
 * This class implements a lightweight bag-of-words database optimized for
 * binary descriptors (BRIEF). It uses hierarchical vocabulary clustering
 * and efficient similarity scoring for real-time loop closure detection.
 * The implementation is adapted from DBoW2 but optimized for memory usage
 * and computational efficiency on resource-constrained devices.
 */
class BowDatabase {

public:
  /**
   * @brief Constructor
   * @param vocabulary_path Path to binary vocabulary file
   */
  explicit BowDatabase(const std::string& vocabulary_path = "");

  /**
   * @brief Initialize database with vocabulary file
   * @param vocabulary_path Path to binary vocabulary file
   * @return true if successful, false otherwise
   */
  bool initialize(const std::string& vocabulary_path);

  /**
   * @brief Add keyframe to database
   * @param keyframe_id Unique identifier for keyframe
   * @param descriptors BRIEF descriptors for the keyframe
   * @return true if successful, false otherwise
   */
  bool addKeyframe(int keyframe_id, const std::vector<BriefDescriptor>& descriptors);

  /**
   * @brief Query database for similar keyframes
   * @param descriptors Query descriptors
   * @param candidates Output vector of candidate matches
   * @param max_candidates Maximum number of candidates to return
   * @return Number of candidates found
   */
  int query(const std::vector<BriefDescriptor>& descriptors,
            std::vector<LoopCandidate>& candidates,
            int max_candidates = 5);

  /**
   * @brief Remove old keyframes from database (for memory management)
   * @param keyframe_ids Vector of keyframe IDs to remove
   */
  void removeKeyframes(const std::vector<int>& keyframe_ids);

  /**
   * @brief Clear entire database
   */
  void clear();

  /**
   * @brief Get number of keyframes in database
   * @return Number of keyframes
   */
  size_t size() const { return keyframe_data_.size(); }

  /**
   * @brief Check if database is initialized
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const { return is_initialized_; }

private:
  struct VocabularyNode {
    int id;
    BriefDescriptor descriptor;
    double weight;
    std::vector<int> children;
    bool is_leaf;

    VocabularyNode() : id(-1), weight(0.0), is_leaf(true) {}
  };

  struct KeyframeBoW {
    int keyframe_id;
    std::map<int, double> bow_vector; // word_id -> weight
    double norm;

    KeyframeBoW() : keyframe_id(-1), norm(0.0) {}
  };

  bool is_initialized_;
  std::vector<VocabularyNode> vocabulary_;
  std::map<int, KeyframeBoW> keyframe_data_;
  std::unordered_map<int, std::vector<int>> inverted_index_; // word_id -> keyframe_ids

  /**
   * @brief Load vocabulary from binary file
   * @param vocabulary_path Path to vocabulary file
   * @return true if successful, false otherwise
   */
  bool loadVocabulary(const std::string& vocabulary_path);

  /**
   * @brief Create default vocabulary if file not available
   */
  void createDefaultVocabulary();

  /**
   * @brief Transform descriptors to bag-of-words representation
   * @param descriptors Input descriptors
   * @param bow_vector Output BoW vector
   */
  void transformToBow(const std::vector<BriefDescriptor>& descriptors,
                      std::map<int, double>& bow_vector);

  /**
   * @brief Find closest vocabulary word for descriptor
   * @param descriptor Input descriptor
   * @return Word ID of closest vocabulary word
   */
  int findClosestWord(const BriefDescriptor& descriptor);

  /**
   * @brief Compute similarity score between two BoW vectors
   * @param bow1 First BoW vector
   * @param bow2 Second BoW vector
   * @return Similarity score [0, 1]
   */
  double computeSimilarity(const std::map<int, double>& bow1,
                          const std::map<int, double>& bow2);

  /**
   * @brief Normalize BoW vector
   * @param bow_vector Input/output BoW vector
   * @return L2 norm of the vector
   */
  double normalizeBowVector(std::map<int, double>& bow_vector);

  /**
   * @brief Update inverted index for new keyframe
   * @param keyframe_id Keyframe ID
   * @param bow_vector BoW vector for the keyframe
   */
  void updateInvertedIndex(int keyframe_id, const std::map<int, double>& bow_vector);

  /**
   * @brief Remove keyframe from inverted index
   * @param keyframe_id Keyframe ID to remove
   */
  void removeFromInvertedIndex(int keyframe_id);

  /**
   * @brief Get candidate keyframes using inverted index
   * @param bow_vector Query BoW vector
   * @param candidate_keyframes Output candidate keyframe IDs
   */
  void getCandidateKeyframes(const std::map<int, double>& bow_vector,
                            std::vector<int>& candidate_keyframes);
};

} // namespace ov_msckf

#endif // OV_MSCKF_BOW_DATABASE_H