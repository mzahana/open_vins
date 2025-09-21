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
#include <random>
#include <vector>

#include "loop/BowDatabase.h"

using namespace ov_msckf;

class BowDatabaseTest : public ::testing::Test {
protected:
  void SetUp() override {
    database = std::make_unique<BowDatabase>();

    // Create some test descriptors
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (int i = 0; i < num_test_descriptors; ++i) {
      BriefDescriptor desc;
      for (int bit = 0; bit < 256; ++bit) {
        if (bit_dist(rng)) {
          desc[bit] = 1;
        }
      }
      test_descriptors.push_back(desc);
    }
  }

  std::unique_ptr<BowDatabase> database;
  std::vector<BriefDescriptor> test_descriptors;
  static const int num_test_descriptors = 100;
};

TEST_F(BowDatabaseTest, ConstructorTest) {
  EXPECT_TRUE(database != nullptr);

  // Should not be initialized without vocabulary
  EXPECT_FALSE(database->isInitialized());

  // Should have zero size
  EXPECT_EQ(database->size(), 0);
}

TEST_F(BowDatabaseTest, InitializationTest) {
  // Initialize with empty path (should create default vocabulary)
  EXPECT_TRUE(database->initialize(""));
  EXPECT_TRUE(database->isInitialized());

  // Test with invalid path
  BowDatabase database2;
  EXPECT_TRUE(database2.initialize("/invalid/path/to/vocab.bin")); // Should fallback to default
  EXPECT_TRUE(database2.isInitialized());
}

TEST_F(BowDatabaseTest, AddKeyframeTest) {
  database->initialize("");

  // Add keyframes with test descriptors
  for (int i = 0; i < 10; ++i) {
    std::vector<BriefDescriptor> frame_descriptors;
    frame_descriptors.insert(frame_descriptors.end(),
                           test_descriptors.begin() + i * 10,
                           test_descriptors.begin() + (i + 1) * 10);

    bool success = database->addKeyframe(i, frame_descriptors);
    EXPECT_TRUE(success);
  }

  EXPECT_EQ(database->size(), 10);
}

TEST_F(BowDatabaseTest, AddKeyframeEdgeCases) {
  database->initialize("");

  // Add empty descriptors
  std::vector<BriefDescriptor> empty_descriptors;
  bool success = database->addKeyframe(0, empty_descriptors);
  EXPECT_FALSE(success); // Should fail with empty descriptors

  // Add with uninitialized database
  BowDatabase uninit_database;
  success = uninit_database.addKeyframe(0, test_descriptors);
  EXPECT_FALSE(success);
}

TEST_F(BowDatabaseTest, QueryTest) {
  database->initialize("");

  // Add some keyframes
  for (int i = 0; i < 5; ++i) {
    std::vector<BriefDescriptor> frame_descriptors;
    frame_descriptors.insert(frame_descriptors.end(),
                           test_descriptors.begin() + i * 20,
                           test_descriptors.begin() + (i + 1) * 20);

    database->addKeyframe(i, frame_descriptors);
  }

  // Query with similar descriptors (should find matches)
  std::vector<BriefDescriptor> query_descriptors;
  query_descriptors.insert(query_descriptors.end(),
                         test_descriptors.begin(),
                         test_descriptors.begin() + 20);

  std::vector<LoopCandidate> candidates;
  int num_candidates = database->query(query_descriptors, candidates, 3);

  EXPECT_GT(num_candidates, 0);
  EXPECT_LE(num_candidates, 3);
  EXPECT_EQ(static_cast<int>(candidates.size()), num_candidates);

  // Check candidate validity
  for (const auto& candidate : candidates) {
    EXPECT_GE(candidate.match_keyframe_id, 0);
    EXPECT_LT(candidate.match_keyframe_id, 5);
    EXPECT_GE(candidate.similarity_score, 0.0);
    EXPECT_LE(candidate.similarity_score, 1.0);
  }
}

TEST_F(BowDatabaseTest, QueryEmptyTest) {
  database->initialize("");

  std::vector<LoopCandidate> candidates;

  // Query empty database
  int num_candidates = database->query(test_descriptors, candidates, 5);
  EXPECT_EQ(num_candidates, 0);
  EXPECT_TRUE(candidates.empty());

  // Query with empty descriptors
  std::vector<BriefDescriptor> empty_descriptors;
  num_candidates = database->query(empty_descriptors, candidates, 5);
  EXPECT_EQ(num_candidates, 0);
  EXPECT_TRUE(candidates.empty());
}

TEST_F(BowDatabaseTest, RemoveKeyframesTest) {
  database->initialize("");

  // Add keyframes
  for (int i = 0; i < 10; ++i) {
    std::vector<BriefDescriptor> frame_descriptors;
    frame_descriptors.insert(frame_descriptors.end(),
                           test_descriptors.begin() + i * 10,
                           test_descriptors.begin() + (i + 1) * 10);

    database->addKeyframe(i, frame_descriptors);
  }

  EXPECT_EQ(database->size(), 10);

  // Remove some keyframes
  std::vector<int> to_remove = {1, 3, 5, 7};
  database->removeKeyframes(to_remove);

  EXPECT_EQ(database->size(), 6);

  // Query should not return removed keyframes
  std::vector<LoopCandidate> candidates;
  int num_candidates = database->query(test_descriptors, candidates, 10);

  for (const auto& candidate : candidates) {
    EXPECT_TRUE(std::find(to_remove.begin(), to_remove.end(),
                         candidate.match_keyframe_id) == to_remove.end());
  }
}

TEST_F(BowDatabaseTest, ClearTest) {
  database->initialize("");

  // Add keyframes
  for (int i = 0; i < 5; ++i) {
    std::vector<BriefDescriptor> frame_descriptors(10, test_descriptors[0]);
    database->addKeyframe(i, frame_descriptors);
  }

  EXPECT_EQ(database->size(), 5);

  // Clear database
  database->clear();
  EXPECT_EQ(database->size(), 0);

  // Query should return no results
  std::vector<LoopCandidate> candidates;
  int num_candidates = database->query(test_descriptors, candidates, 5);
  EXPECT_EQ(num_candidates, 0);
}

TEST_F(BowDatabaseTest, SimilarityConsistencyTest) {
  database->initialize("");

  // Add identical keyframe twice with different IDs
  std::vector<BriefDescriptor> frame_descriptors;
  frame_descriptors.insert(frame_descriptors.end(),
                         test_descriptors.begin(),
                         test_descriptors.begin() + 50);

  database->addKeyframe(0, frame_descriptors);
  database->addKeyframe(1, frame_descriptors);

  // Query with same descriptors
  std::vector<LoopCandidate> candidates;
  int num_candidates = database->query(frame_descriptors, candidates, 5);

  EXPECT_GE(num_candidates, 2);

  // Both keyframes should have similar (high) similarity scores
  double max_score = 0.0;
  for (const auto& candidate : candidates) {
    max_score = std::max(max_score, candidate.similarity_score);
  }

  // Should have high similarity for identical descriptors
  EXPECT_GT(max_score, 0.5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}