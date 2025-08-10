#pragma once

#include <arma.hpp>
#include <Eigen/Dense>
#include <alias.hpp>
#include <bip.hpp>
#include <robust.hpp>

namespace robarma::s
{
    struct cost
    {
    private:
        arma_model model;

    public:
        cost(arma_model model)
            : model(model)
        {
        }

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);
            // Set delta as delta = max rho1 / 2
            // Max of rho1 = 3.25
            T delta = T(3.25 / 2);
            std::function<Vec<T>(Vec<T>)> func = static_cast<Vec<T> (*)(const Vec<T>)>(&robarma::bip::rho1);
            T est = robarma::base::scale(model.arma_residuals(phi, theta, mu), delta, func);
            residuals[0] = est;
            return true;
        };
    };
}
// end of file