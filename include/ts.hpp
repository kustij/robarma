/**
 * @file ts.hpp
 * @brief General time series utilities needed in time series analysis.
 *
 */
#pragma once

#include <Eigen/Dense>
#include <arma.hpp>
#include <alias.hpp>
#include <robust.hpp>

namespace robarma
{

    template <typename T>
    inline Mat<T> robust_autocov_matrix(const Vec<T> &y, const int &m, const int &n)
    {
        int N = y.size();
        T med = robarma::base::median(y);
        Vec<T> yc = y.array() - med;
        Vec<T> psi = robarma::base::huber(yc);

        Mat<T> a = Mat<T>::Zero(m, n);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int h = std::abs(i - j);
                if (N - h <= 0)
                {
                    a(i, j) = T(0);
                    continue;
                }
                // Use Eigen blocks for fast segment access
                auto sub = psi.segment(0, N - h);
                auto lag = psi.segment(h, N - h);
                a(i, j) = sub.cwiseProduct(lag).sum() / T(N - h);
            }
        }
        return a;
    }

    // Sample autocovariance matrix of size m x n
    template <typename T>
    inline Mat<T> autocov_matrix(const Vec<T> &y, const int &m, const int &n)
    {
        int N = y.size();
        T avg = y.mean();
        Vec<T> yc = y.array() - avg;

        Mat<T> a = Mat<T>::Zero(m, n);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int h = std::abs(i - j);
                if (N - h <= 0)
                {
                    a(i, j) = T(0);
                    continue;
                }
                auto sub = yc.segment(0, N - h);
                auto lag = yc.segment(h, N - h);
                a(i, j) = sub.cwiseProduct(lag).sum() / T(N - h);
            }
        }
        return a;
    }

    template <typename T>
    inline Vec<T> causal(Vec<T> phi, Vec<T> theta)
    {
        // Calculate coefficients for characteristic polynomial of causal representation
        // lambda(B) = theta(B) / phi(B)
        int n = 100;
        int p = phi.size();
        int q = theta.size();
        int k = p + 1;

        Vec<T> lambda = Vec<T>::Zero(n + p);
        Vec<T> ma = Vec<T>::Zero(n + p);

        lambda(p) = T(1);

        if (q > 0)
            ma.segment(k, q) = theta;

        for (size_t i = k; i < n; i++)
            lambda(i) = phi.dot(lambda.segment(i - p, p).reverse()) - ma(i);

        // Omit the first index (index 0 in mathematical notation) that is always 1
        return lambda.tail(n - 1);
    }

}

// end of file