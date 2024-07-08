cmake_minimum_required(VERSION 3.10)
project(micppgrad)

# Set C++ standard to 23 and make it required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(include)

# Main executable
add_executable(micppgrad
  src/main.cpp
  src/value.cpp
  src/neural_network.cpp
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(
  micppgrad_test
  tests/test_value.cpp
  src/value.cpp
)

target_link_libraries(
  micppgrad_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(micppgrad_test)

