#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <alias.hpp>

/**
 * @brief Rho functions used in MM-, BIP-MM and S-estimators as defined in \cite Muler
 *
 */
namespace robarma::bip
{
    template <typename T>
    T eta(const T x)
    {
        if (ceres::abs(x) <= T(2))
        {
            return x;
        }
        else if (T(2) < ceres::abs(x) && ceres::abs(x) <= T(3))
        {
            return T(0.016) * ceres::pow(x, 7) - T(0.312) * ceres::pow(x, 5) + T(1.728) * pow(x, 3) - T(1.944) * x;
        }
        else
        {
            return T(0);
        }
    }

    template <typename T>
    T rho2(const T x)
    {
        if (ceres::abs(x) <= T(2))
        {
            return T(0.5) * ceres::pow(x, 2);
        }
        else if (T(2) < ceres::abs(x) && ceres::abs(x) <= T(3))
        {
            return T(0.002) * ceres::pow(x, 8) - T(0.052) * ceres::pow(x, 6) + T(0.432) * ceres::pow(x, 4) - T(0.972) * ceres::pow(x, 2) + T(1.792);
        }
        else
        {
            return T(3.25);
        }
    }

    template <typename T>
    T rho1(const T x)
    {
        return rho2(x / T(0.405));
    }

    template <typename T>
    Vec<T> rho1(const Vec<T> x)
    {
        return x.unaryExpr(static_cast<T (*)(const T)>(&rho1));
    }

    template <typename T>
    Vec<T> rho2(const Vec<T> x)
    {
        return x.unaryExpr(static_cast<T (*)(const T)>(&rho2));
    }

    template <typename T>
    Vec<T> eta(const Vec<T> x)
    {
        return x.unaryExpr(static_cast<T (*)(const T)>(&eta));
    }
}
// end of file