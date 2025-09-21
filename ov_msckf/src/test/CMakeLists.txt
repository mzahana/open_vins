# OpenVINS Loop Closure Unit Tests
cmake_minimum_required(VERSION 3.8)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Find GTest
find_package(ament_cmake_gtest REQUIRED)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Test executables
if(BUILD_TESTING)
  # BRIEF Extractor Tests
  ament_add_gtest(test_brief_extractor test_brief_extractor.cpp
    ../loop/BriefExtractor.cpp
  )

  target_link_libraries(test_brief_extractor
    ${OpenCV_LIBRARIES}
  )

  # BoW Database Tests
  ament_add_gtest(test_bow_database test_bow_database.cpp
    ../loop/BowDatabase.cpp
    ../loop/BriefExtractor.cpp
  )

  target_link_libraries(test_bow_database
    ${OpenCV_LIBRARIES}
  )

  # Loop Detector Tests
  ament_add_gtest(test_loop_detector test_loop_detector.cpp
    ../loop/LoopDetector.cpp
    ../loop/BowDatabase.cpp
    ../loop/BriefExtractor.cpp
  )

  target_link_libraries(test_loop_detector
    ${OpenCV_LIBRARIES}
  )

  # Loop Updater Tests
  ament_add_gtest(test_updater_loop test_updater_loop.cpp
    ../update/UpdaterLoop.cpp
    ../state/State.cpp
    ../state/StateHelper.cpp
  )

  target_link_libraries(test_updater_loop
    ${OpenCV_LIBRARIES}
  )

  # Set target properties
  set_target_properties(test_brief_extractor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  set_target_properties(test_bow_database PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  set_target_properties(test_loop_detector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )

  set_target_properties(test_updater_loop PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
endif()