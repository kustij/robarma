#pragma once

#include <Eigen/Dense>
#include <alias.hpp>

namespace robarma::base
{
    template <typename Derived>
    inline typename Derived::Scalar median(Eigen::DenseBase<Derived> &d)
    {
        auto r{d.reshaped()};
        std::sort(r.begin(), r.end());
        return r.size() % 2 == 0 ? r.segment((r.size() - 2) / 2, 2).mean() : r(r.size() / 2);
    }

    template <typename Derived>
    inline typename Derived::Scalar median(const Eigen::DenseBase<Derived> &d)
    {
        typename Derived::PlainObject m{d.replicate(1, 1)};
        return median(m);
    }

    // Median Absolute deviation
    template <typename T>
    inline T MAD(const Vec<T> &x)
    {

        T med = median(x);
        T mad = median((x.array() - med).abs());
        return mad;
    }

    // Normalized MAD
    template <typename T>
    inline T MADN(const Vec<T> &x)
    {
        return MAD(x) / T(0.675);
    }

    // Huber psi-function for an Eigen vector
    template <typename T>
    inline Vec<T> huber(const Vec<T> &x, T k = T(1.345))
    {
        return x.unaryExpr([k](const T &xi)
                           {
        if (ceres::abs(xi) <= k)
            return xi;
        else
            return k * ((xi > T(0)) ? T(1) : T(-1)); });
    }

    template <typename T>
    inline T bisquare(const T x, T k = T(1.547645))
    {
        // eff 95%, k = 4.685 for location
        // maximum breakdown of 50%, k = 1.547645 for scale
        if (ceres::abs(x) <= k)
        {
            return T(1) - ceres::pow(T(1) - pow((x / k), 2), 3);
        }
        else
        {
            return T(1);
        }
    }

    template <typename T>
    inline Vec<T> bisquare(const Vec<T> x, const T k)
    {
        return x.unaryExpr([k](const T &xi)
                           { return bisquare(xi, k); });
    }

    template <typename T>
    inline T scale(const Vec<T> x, const T b = T(0.5), std::function<Vec<T>(Vec<T>)> func = [](const Vec<T> &v)
                                                       { return bisquare(v, T(1.547645)); })
    {
        T tol = T(1e-6);
        T err = T(1) + tol;
        int max = 100;
        int i = 0;

        // Assume to be centered.
        // Scale with 0.6745 to get sigma_0 = MADN(x)
        T sigma_0 = median(x.array().abs()) / T(0.6745);
        T sigma_1;

        while ((err > tol) && (i < max))
        {
            i = i + 1;
            sigma_1 = ceres::sqrt(pow(sigma_0, 2) * func((x / sigma_0).eval()).mean() / b);
            err = ceres::abs(sigma_1 - sigma_0) / sigma_0;
            sigma_0 = sigma_1;
        }
        return sigma_0;
    }
}

// end of file