#pragma once

#include <arma.hpp>
#include <alias.hpp>
#include <state_space_cost.hpp>

namespace robarma::mle
{

    struct cost : public robarma::state_space_cost
    {
    public:
        cost(arma_model model)
            : state_space_cost(model)
        {
        }

        template <typename T>
        void predict(Vec<T> &a, Mat<T> &P, const Mat<T> F, const Vec<T> H, const Vec<T> c) const
        {
            a = F * a.template cast<T>() + c;
            P = F * P * F.transpose().template cast<T>() + H * H.transpose().template cast<T>();
        }

        template <typename T>
        void update(Vec<T> &a, Mat<T> &P, const T v, const T f, const Vec<T> &z) const
        {
            a = a + P * z * v / f;
            T epsilon = std::numeric_limits<T>::epsilon();
            P = P - (P * z * z.transpose() * P) / (f + epsilon);
        }

        template <typename T>
        T loss(Vec<T> w, Vec<T> f) const
        {
            T sigma = w.array().square().mean();
            T vdet = f.prod();
            return (T)model.n * log(sigma) + log(vdet);
        }

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);

            Vec<T> z = Vec<T>::Zero(r);
            z.head(1).setOnes();

            Mat<T> F = F0(phi);
            Vec<T> H = H0(theta);
            Mat<T> P = P0(F, H);

            Vec<T> f = Vec<T>::Ones(model.n);
            Vec<T> v = Vec<T>::Zero(model.n);
            Vec<T> w = Vec<T>::Zero(model.n);

            Vec<T> a = Vec<T>::Zero(r);
            Vec<T> c = c0(phi, mu);

            for (size_t i = 0; i < model.n; i++)
            {
                predict(a, P, F, H, c);
                f(i) = T(z.transpose() * P * z);
                v(i) = T(model.y(i)) - T(z.transpose() * a);
                w(i) = v(i) / ceres::sqrt(f(i));
                update(a, P, v(i), f(i), z);
            }
            residuals[0] = loss(w, f);
            return true;
        };
    };

}
// end of file
