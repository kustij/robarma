#pragma once

#include <Eigen/Dense>

/**
 * @brief Aliases for templated Eigen classes used in the project.
 *
 * @tparam T
 */

template <class T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <class T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;