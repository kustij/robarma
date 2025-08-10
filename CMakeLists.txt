cmake_minimum_required(VERSION 3.16)
project(robarma VERSION 1.0 LANGUAGES CXX)


# Dependencies
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Catch2 3 REQUIRED)

# Header-only library target
add_library(robarma INTERFACE)
target_compile_features(robarma INTERFACE cxx_std_17)
target_include_directories(robarma INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(robarma INTERFACE -march=native -O3 -fno-math-errno -fno-trapping-math)
target_link_libraries(robarma INTERFACE Eigen3::Eigen Ceres::ceres)

# Tests
add_executable(robarma_tests tests/test_main.cpp)
target_link_libraries(robarma_tests PRIVATE robarma Catch2::Catch2WithMain)
enable_testing()
add_test(NAME robarma_tests COMMAND robarma_tests)

# Install rules (optional)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS robarma EXPORT robarmaTargets)