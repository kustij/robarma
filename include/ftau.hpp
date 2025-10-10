#pragma once

#include <alias.hpp>
#include <arma.hpp>
#include <state_space_cost.hpp>
#include <tau.hpp>

namespace robarma::ftau
{
    struct cost : public robarma::state_space_cost
    {
        cost(arma_model model)
            : state_space_cost(model)
        {
        }

        template <typename T>
        void predict(Vec<T> &a, Mat<T> &P, const Mat<T> F, const Vec<T> H, const T sigma, const Vec<T> c) const
        {
            a = (F * a) + c;
            P = (F * P * F.transpose()) + (pow(sigma, 2) * H * H.transpose());
        }

        template <typename T>
        void update(Vec<T> &a, Mat<T> &P, const T u, const T s, const Vec<T> mt) const
        {
            a = a + ((mt / s) * tau::psi(u / s));
            P = P - (mt * mt.transpose() / pow(s, 2) * tau::w(u / s));
        }

        template <typename T>
        T loss(Vec<T> u, Vec<T> a) const
        {
            T S = tau::tau2<T>(u.array() / a.array());
            T log_likelihood = (T)model.n * log(S) + a.array().square().log().sum();
            return log_likelihood;
        }

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);

            // Fix the estimate of sigma as the centered time series
            Vec<T> y_centered = model.y.template cast<T>().array() - T(base::median(model.y));
            T sigma = robarma::tau::s<T>(y_centered);

            Vec<T> z = Vec<T>::Zero(r);
            z.head(1).setOnes();

            Mat<T> F = F0(phi);
            Vec<T> H = H0(theta);
            Mat<T> P = robust_autocov_matrix<T>(model.y.template cast<T>(), r, r);

            Vec<T> s = Vec<T>::Ones(model.n);
            Vec<T> u = Vec<T>::Zero(model.n);
            Vec<T> mt = P.col(0);

            Vec<T> a = Vec<T>::Zero(r);
            Vec<T> c = c0(phi, mu);

            for (int i = 1; i < model.n; i++)
            {
                predict(a, P, F, H, sigma, c);
                mt = P.col(0);
                s(i) = ceres::sqrt(mt(0));
                u(i) = T(model.y(i)) - T(z.transpose() * a);
                update(a, P, u(i), s(i), mt);
            }
            residuals[0] = loss(u, (s / sigma).eval());
            return true;
        }
    };
} // namespace robarma::ftau
// end of file
