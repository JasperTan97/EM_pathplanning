cmake_minimum_required(VERSION 3.5)
project(em_vehicle_control_msgs)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Declare the message files to be processed
set(msg_files
  "msg/Pose2D.msg"
  "msg/Path2D.msg"
)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose2D.msg"
  "msg/Path2D.msg"
  DEPENDENCIES std_msgs
)

# Ensure that message generation happens before Python setup
ament_export_dependencies(rosidl_default_runtime)
ament_package()
