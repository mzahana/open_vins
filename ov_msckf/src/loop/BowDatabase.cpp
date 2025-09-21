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

#include "BowDatabase.h"
#include "BriefExtractor.h"
#include "utils/print.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

using namespace ov_msckf;

BowDatabase::BowDatabase(const std::string& vocabulary_path) : is_initialized_(false) {
  if (!vocabulary_path.empty()) {
    initialize(vocabulary_path);
  }
}

bool BowDatabase::initialize(const std::string& vocabulary_path) {
  if (!vocabulary_path.empty() && loadVocabulary(vocabulary_path)) {
    is_initialized_ = true;
    PRINT_INFO("[BOW]: Loaded vocabulary from %s\n", vocabulary_path.c_str());
  } else {
    PRINT_WARNING("[BOW]: Could not load vocabulary, creating default vocabulary\n");
    createDefaultVocabulary();
    is_initialized_ = true;
  }
  return is_initialized_;
}

bool BowDatabase::loadVocabulary(const std::string& vocabulary_path) {
  std::ifstream file(vocabulary_path, std::ios::binary);
  if (!file.is_open()) {
    PRINT_ERROR("[BOW]: Cannot open vocabulary file: %s\n", vocabulary_path.c_str());
    return false;
  }

  try {
    // Read number of nodes
    int num_nodes;
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));

    if (num_nodes <= 0 || num_nodes > 1000000) { // Sanity check
      PRINT_ERROR("[BOW]: Invalid vocabulary file format\n");
      return false;
    }

    vocabulary_.clear();
    vocabulary_.reserve(num_nodes);

    // Read vocabulary nodes
    for (int i = 0; i < num_nodes; ++i) {
      VocabularyNode node;

      // Read node ID
      file.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));

      // Read descriptor (256 bits = 32 bytes)
      std::string desc_str(32, '\0');
      file.read(&desc_str[0], 32);
      node.descriptor = BriefDescriptor(desc_str);

      // Read weight
      file.read(reinterpret_cast<char*>(&node.weight), sizeof(node.weight));

      // Read number of children
      int num_children;
      file.read(reinterpret_cast<char*>(&num_children), sizeof(num_children));

      // Read children IDs
      node.children.resize(num_children);
      for (int j = 0; j < num_children; ++j) {
        file.read(reinterpret_cast<char*>(&node.children[j]), sizeof(node.children[j]));
      }

      // Read is_leaf flag
      file.read(reinterpret_cast<char*>(&node.is_leaf), sizeof(node.is_leaf));

      vocabulary_.push_back(node);
    }

    PRINT_INFO("[BOW]: Loaded vocabulary with %d nodes\n", num_nodes);
    return true;

  } catch (const std::exception& e) {
    PRINT_ERROR("[BOW]: Error reading vocabulary file: %s\n", e.what());
    return false;
  }
}

void BowDatabase::createDefaultVocabulary() {
  // Create a simple default vocabulary with random descriptors
  const int num_words = 1000;
  vocabulary_.clear();
  vocabulary_.reserve(num_words);

  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_int_distribution<int> bit_dist(0, 1);

  for (int i = 0; i < num_words; ++i) {
    VocabularyNode node;
    node.id = i;
    node.weight = 1.0 / num_words; // Equal weights
    node.is_leaf = true;

    // Generate random descriptor
    for (int bit = 0; bit < 256; ++bit) {
      if (bit_dist(rng)) {
        node.descriptor[bit] = 1;
      }
    }

    vocabulary_.push_back(node);
  }

  PRINT_INFO("[BOW]: Created default vocabulary with %d words\n", num_words);
}

bool BowDatabase::addKeyframe(int keyframe_id, const std::vector<BriefDescriptor>& descriptors) {
  PRINT_DEBUG("[BOW]: Adding keyframe %d with %zu descriptors\n", keyframe_id, descriptors.size());

  if (!is_initialized_ || descriptors.empty()) {
    PRINT_DEBUG("[BOW]: Add keyframe failed - not initialized (%s) or no descriptors (%s)\n",
                is_initialized_ ? "true" : "false",
                descriptors.empty() ? "true" : "false");
    return false;
  }

  KeyframeBoW keyframe_bow;
  keyframe_bow.keyframe_id = keyframe_id;

  // Transform descriptors to BoW representation
  transformToBow(descriptors, keyframe_bow.bow_vector);

  // Normalize BoW vector
  keyframe_bow.norm = normalizeBowVector(keyframe_bow.bow_vector);

  PRINT_DEBUG("[BOW]: Keyframe %d BoW vector size: %zu, norm: %.6f\n",
              keyframe_id, keyframe_bow.bow_vector.size(), keyframe_bow.norm);

  // Store keyframe data
  keyframe_data_[keyframe_id] = keyframe_bow;

  // Update inverted index
  updateInvertedIndex(keyframe_id, keyframe_bow.bow_vector);

  PRINT_DEBUG("[BOW]: Successfully added keyframe %d, total keyframes: %zu\n",
              keyframe_id, keyframe_data_.size());

  return true;
}

int BowDatabase::query(const std::vector<BriefDescriptor>& descriptors,
                      std::vector<LoopCandidate>& candidates,
                      int max_candidates) {
  candidates.clear();

  PRINT_DEBUG("[BOW]: Starting query with %zu descriptors\n", descriptors.size());
  PRINT_DEBUG("[BOW]: Total keyframes in database: %zu\n", keyframe_data_.size());
  PRINT_DEBUG("[BOW]: Vocabulary size: %zu\n", vocabulary_.size());

  if (!is_initialized_ || descriptors.empty()) {
    PRINT_DEBUG("[BOW]: Query failed - not initialized (%s) or no descriptors (%s)\n",
                is_initialized_ ? "true" : "false",
                descriptors.empty() ? "true" : "false");
    return 0;
  }

  // Transform query descriptors to BoW
  std::map<int, double> query_bow;
  transformToBow(descriptors, query_bow);
  double query_norm = normalizeBowVector(query_bow);

  PRINT_DEBUG("[BOW]: Query BoW vector size: %zu, norm: %.6f\n", query_bow.size(), query_norm);

  if (query_norm == 0.0) {
    PRINT_DEBUG("[BOW]: Query failed - zero norm\n");
    return 0;
  }

  // Get candidate keyframes using inverted index
  std::vector<int> candidate_keyframes;
  getCandidateKeyframes(query_bow, candidate_keyframes);

  PRINT_DEBUG("[BOW]: Found %zu candidate keyframes from inverted index\n", candidate_keyframes.size());

  // Compute similarity scores for candidates
  std::vector<std::pair<double, int>> scored_candidates;
  scored_candidates.reserve(candidate_keyframes.size());

  for (int keyframe_id : candidate_keyframes) {
    auto it = keyframe_data_.find(keyframe_id);
    if (it != keyframe_data_.end()) {
      double similarity = computeSimilarity(query_bow, it->second.bow_vector);
      PRINT_DEBUG("[BOW]: Keyframe %d similarity: %.6f\n", keyframe_id, similarity);
      if (similarity > 0.0) {
        scored_candidates.emplace_back(similarity, keyframe_id);
      }
    }
  }

  PRINT_DEBUG("[BOW]: %zu candidates with positive similarity\n", scored_candidates.size());

  // Sort by similarity score (descending)
  std::sort(scored_candidates.begin(), scored_candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  // Convert to LoopCandidate format
  int num_candidates = std::min(max_candidates, static_cast<int>(scored_candidates.size()));
  candidates.reserve(num_candidates);

  for (int i = 0; i < num_candidates; ++i) {
    LoopCandidate candidate;
    candidate.match_keyframe_id = scored_candidates[i].second;
    candidate.similarity_score = scored_candidates[i].first;
    candidates.push_back(candidate);
    PRINT_DEBUG("[BOW]: Final candidate %d: keyframe %d, similarity %.6f\n",
                i, candidate.match_keyframe_id, candidate.similarity_score);
  }

  PRINT_DEBUG("[BOW]: Returning %d candidates\n", num_candidates);
  return num_candidates;
}

void BowDatabase::transformToBow(const std::vector<BriefDescriptor>& descriptors,
                                std::map<int, double>& bow_vector) {
  bow_vector.clear();

  if (vocabulary_.empty()) {
    return;
  }

  // Count occurrences of each word
  std::map<int, int> word_counts;
  for (const auto& descriptor : descriptors) {
    int word_id = findClosestWord(descriptor);
    if (word_id >= 0) {
      word_counts[word_id]++;
    }
  }

  // Convert to TF-IDF weights
  double total_descriptors = static_cast<double>(descriptors.size());
  for (const auto& word_count : word_counts) {
    int word_id = word_count.first;
    double tf = word_count.second / total_descriptors; // Term frequency
    double idf = vocabulary_[word_id].weight; // Inverse document frequency (pre-computed)
    bow_vector[word_id] = tf * idf;
  }
}

int BowDatabase::findClosestWord(const BriefDescriptor& descriptor) {
  if (vocabulary_.empty()) {
    return -1;
  }

  int best_word_id = -1;
  int min_distance = std::numeric_limits<int>::max();

  for (size_t i = 0; i < vocabulary_.size(); ++i) {
    int distance = BriefExtractor::hammingDistance(descriptor, vocabulary_[i].descriptor);
    if (distance < min_distance) {
      min_distance = distance;
      best_word_id = static_cast<int>(i);
    }
  }

  return best_word_id;
}

double BowDatabase::computeSimilarity(const std::map<int, double>& bow1,
                                     const std::map<int, double>& bow2) {
  if (bow1.empty() || bow2.empty()) {
    return 0.0;
  }

  // Compute cosine similarity
  double dot_product = 0.0;
  double norm1_sq = 0.0;
  double norm2_sq = 0.0;

  // Compute dot product and norm1
  for (const auto& word_weight1 : bow1) {
    int word_id = word_weight1.first;
    double weight1 = word_weight1.second;
    norm1_sq += weight1 * weight1;

    auto it2 = bow2.find(word_id);
    if (it2 != bow2.end()) {
      dot_product += weight1 * it2->second;
    }
  }

  // Compute norm2
  for (const auto& word_weight2 : bow2) {
    double weight2 = word_weight2.second;
    norm2_sq += weight2 * weight2;
  }

  // Avoid division by zero
  double norm_product = std::sqrt(norm1_sq * norm2_sq);
  if (norm_product < 1e-12) {
    return 0.0;
  }

  return dot_product / norm_product;
}

double BowDatabase::normalizeBowVector(std::map<int, double>& bow_vector) {
  double norm_sq = 0.0;
  for (const auto& word_weight : bow_vector) {
    norm_sq += word_weight.second * word_weight.second;
  }

  double norm = std::sqrt(norm_sq);
  if (norm > 1e-12) {
    for (auto& word_weight : bow_vector) {
      word_weight.second /= norm;
    }
  }

  return norm;
}

void BowDatabase::updateInvertedIndex(int keyframe_id, const std::map<int, double>& bow_vector) {
  for (const auto& word_weight : bow_vector) {
    int word_id = word_weight.first;
    inverted_index_[word_id].push_back(keyframe_id);
  }
}

void BowDatabase::removeFromInvertedIndex(int keyframe_id) {
  for (auto& word_keyframes : inverted_index_) {
    auto& keyframes = word_keyframes.second;
    keyframes.erase(std::remove(keyframes.begin(), keyframes.end(), keyframe_id),
                   keyframes.end());
  }
}

void BowDatabase::getCandidateKeyframes(const std::map<int, double>& bow_vector,
                                       std::vector<int>& candidate_keyframes) {
  std::set<int> unique_candidates;

  for (const auto& word_weight : bow_vector) {
    int word_id = word_weight.first;
    auto it = inverted_index_.find(word_id);
    if (it != inverted_index_.end()) {
      for (int keyframe_id : it->second) {
        unique_candidates.insert(keyframe_id);
      }
    }
  }

  candidate_keyframes.assign(unique_candidates.begin(), unique_candidates.end());
}

void BowDatabase::removeKeyframes(const std::vector<int>& keyframe_ids) {
  for (int keyframe_id : keyframe_ids) {
    keyframe_data_.erase(keyframe_id);
    removeFromInvertedIndex(keyframe_id);
  }
}

void BowDatabase::clear() {
  keyframe_data_.clear();
  inverted_index_.clear();
}