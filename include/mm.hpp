#pragma once

#include <arma.hpp>
#include <Eigen/Dense>
#include <alias.hpp>
#include <bip.hpp>
#include <solver.hpp>

namespace robarma::mm
{
    struct cost
    {
    private:
        arma_model model;
        double sigma;

    public:
        cost(arma_model model, double sigma)
            : model(model), sigma(sigma)
        {
        }

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);

            Vec<T> e = model.arma_residuals(phi, theta, mu) / T(sigma);
            T est = robarma::bip::rho2(e).sum() / T(model.n - model.p);
            residuals[0] = est;
            return true;
        };
    };

    arma_fit mm(const arma_model &model, const double &sigma, arma_fit &initial)
    {
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<cost, 4>(new cost(model, sigma));

        ceres::Solver::Options options;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::mm, cost_function, options);

        return fit;
    }
}
// end of file
