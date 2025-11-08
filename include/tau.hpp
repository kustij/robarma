/**
 * @file tau.hpp
 * @brief rho and other functions needed in tau-estimation
 *
 */

#pragma once

#include <Eigen/Dense>
#include <alias.hpp>
#include <robust.hpp>

/**
 * @brief Robust psi and rho functions used in \cite Bianco
 *
 */
namespace robarma::tau
{
    template <typename T>
    inline T rho1(const T x)
    {
        T c = T(1.55);
        T div = x / c;

        if (ceres::abs(x) <= c)
        {
            return T(3) * ceres::pow(div, 2) - T(3) * ceres::pow(div, 4) + ceres::pow(div, 6);
        }
        return T(1);
    }

    template <typename T>
    inline Vec<T> rho1(const Vec<T> x)
    {
        return x.unaryExpr(static_cast<T (*)(const T)>(&rho1));
    }

    template <typename T>
    inline T rho2(const T x)
    {
        T c = T(2.8);

        if (ceres::abs(x) <= c)
        {
            return T(0.14) * ceres::pow(x, 2) + T(0.012) * ceres::pow(x, 4) - T(0.0018) * ceres::pow(x, 6);
        }
        return T(1);
    }

    template <typename T>
    inline Vec<T> rho2(const Vec<T> x)
    {
        return x.unaryExpr(static_cast<T (*)(const T)>(&rho2));
    }

    template <typename T>
    inline T psi(T x)
    {
        // Function psi is a bounded odd function
        T c = T(1.55);
        if (ceres::abs(x) <= c)
        {
            return x;
        }
        return (x > T(0)) ? c : -c;
    }

    template <typename T>
    inline T w(T x)
    {
        if (x == T(0))
        {
            return T(0);
        }
        return psi<T>(x) / x;
    }

    template <typename T>
    inline T s(Vec<T> u)
    {
        // Assume that u is a vector of residuals
        std::function<Vec<T>(Vec<T>)> func = static_cast<Vec<T> (*)(const Vec<T>)>(&rho1);
        return robarma::base::scale(u, T(0.5), func);
    }

    template <typename T>
    inline T tau2(Vec<T> u)
    {
        T sn = s(u);
        return ceres::pow(sn, 2) * rho2((u / sn).eval()).sum();
    }
} // namespace robarma::tau
// end of file