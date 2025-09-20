cmake_minimum_required(VERSION 3.16)
project(robarma VERSION 1.0 LANGUAGES CXX)


# Dependencies
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(Catch2 3 REQUIRED)

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

# Custom 'tests' target for convenience
add_custom_target(tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS robarma_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Build and run all tests"
)

# Install rules (optional)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS robarma EXPORT robarmaTargets)