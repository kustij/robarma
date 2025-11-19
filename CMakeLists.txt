cmake_minimum_required(VERSION 3.16)
project(robarma VERSION 1.0 LANGUAGES CXX)

# Dependencies (cross-platform, vcpkg-friendly)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)

# Header-only library target
if(APPLE)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE CLANG_VERSION)
        string(FIND "${CLANG_VERSION}" "Apple" APPLE_CLANG_FOUND)
        if (APPLE_CLANG_FOUND EQUAL -1)
            message(WARNING "You are building with non-Apple Clang on macOS. This may cause linker/ABI errors. It is recommended to use Apple Clang for all dependencies and your project.")
        endif()
    endif()
endif()

# Header-only INTERFACE target for robarma
add_library(robarma INTERFACE)
target_compile_features(robarma INTERFACE cxx_std_17)
target_include_directories(robarma INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Platform-agnostic optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(robarma INTERFACE -O3 -fno-math-errno -fno-trapping-math)
    if(NOT WIN32)
        target_compile_options(robarma INTERFACE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(robarma INTERFACE /O2)
endif()
target_link_libraries(robarma INTERFACE Eigen3::Eigen Ceres::ceres)


# Option to build tests
option(ROBARMA_BUILD_TESTS "Build robarma tests" ON)
if(ROBARMA_BUILD_TESTS)
    add_executable(robarma_tests tests/test_main.cpp tests/test_numerical_stability.cpp)
    target_link_libraries(robarma_tests PRIVATE robarma Catch2::Catch2WithMain)
    enable_testing()
    add_test(NAME robarma_tests COMMAND robarma_tests)
endif()

# Install rules
install(DIRECTORY include/ DESTINATION include)
install(TARGETS robarma EXPORT robarmaTargets)
install(EXPORT robarmaTargets
    FILE robarmaTargets.cmake
    NAMESPACE robarma::
    DESTINATION lib/cmake/robarma
)