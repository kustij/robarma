#pragma once

#include <arma.hpp>
#include <Eigen/Dense>
#include <alias.hpp>
#include <solver.hpp>

namespace robarma::ols
{
    struct cost
    {
    private:
        arma_model model;

    public:
        cost(arma_model model)
            : model(model) {}

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);

            Vec<T> e = model.arma_residuals(phi, theta, mu);
            residuals[0] = e.array().square().sum();
            return true;
        };
    };
}
// end of file
