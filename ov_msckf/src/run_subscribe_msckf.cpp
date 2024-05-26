#include <memory>
#include <atomic>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"

#if ROS_AVAILABLE == 1
#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#elif ROS_AVAILABLE == 2
#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#endif

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
#if ROS_AVAILABLE == 1
std::shared_ptr<ROS1Visualizer> viz;
std::shared_ptr<ros::NodeHandle> nh;
ros::ServiceServer reset_service;
#elif ROS_AVAILABLE == 2
std::shared_ptr<ROS2Visualizer> viz;
std::shared_ptr<rclcpp::Node> node;
rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service;
#endif

std::atomic<bool> reset_requested(false);

void start_vio_system(const std::string& config_path) {
  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#if ROS_AVAILABLE == 1
  parser->set_node_handler(nh);
#elif ROS_AVAILABLE == 2
  parser->set_node(node);
#endif

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
#if ROS_AVAILABLE == 1
  viz = std::make_shared<ROS1Visualizer>(nh, sys);
  viz->setup_subscribers(parser);
#elif ROS_AVAILABLE == 2
  viz = std::make_shared<ROS2Visualizer>(node, sys);
  viz->setup_subscribers(parser);
#endif

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }
}

#if ROS_AVAILABLE == 1
bool reset_callback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {
#elif ROS_AVAILABLE == 2
void reset_callback(const std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res) {
#endif
  reset_requested.store(true);
#if ROS_AVAILABLE == 1
  return true;
#elif ROS_AVAILABLE == 2
  res->success = true;
#endif
}

int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

#if ROS_AVAILABLE == 1
  // Launch our ros node
  ros::init(argc, argv, "run_subscribe_msckf");
  nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);
  // Start VIO system
  start_vio_system(config_path);
  // Create reset service
  reset_service = nh->advertiseService("reset_vio", reset_callback);
#elif ROS_AVAILABLE == 2
  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  node = std::make_shared<rclcpp::Node>("run_subscribe_msckf", options);
  node->get_parameter<std::string>("config_path", config_path);
  // Start VIO system
  start_vio_system(config_path);
  // Create reset service
  reset_service = node->create_service<std_srvs::srv::Empty>("reset_vio", reset_callback);
#endif

  // Spin off to ROS
  PRINT_DEBUG("done...spinning to ros\n");
#if ROS_AVAILABLE == 1
  ros::AsyncSpinner spinner(0);
  spinner.start();

  while (ros::ok()) {
    if (reset_requested.load()) {
      PRINT_DEBUG("Resetting VIO system...\n");
      start_vio_system(config_path);
      reset_requested.store(false);
    }
    ros::Duration(0.1).sleep();
  }
  ros::waitForShutdown();
#elif ROS_AVAILABLE == 2
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  while (rclcpp::ok()) {
    if (reset_requested.load()) {
      PRINT_DEBUG("Resetting VIO system...\n");
      start_vio_system(config_path);
      reset_requested.store(false);
    }
    executor.spin_some();
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
#endif

  // Final visualization
  viz->visualize_final();
#if ROS_AVAILABLE == 1
  ros::shutdown();
#elif ROS_AVAILABLE == 2
  rclcpp::shutdown();
#endif

  // Done!
  return EXIT_SUCCESS;
}
