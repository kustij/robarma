#pragma once

#include <Eigen/Dense>
#include <alias.hpp>
#include <arma.hpp>
#include <bip.hpp>
#include <solver.hpp>

namespace robarma::bmm
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

            Vec<T> e = model.bip_arma_residuals(phi, theta, mu, T(sigma)) / T(sigma);

            T est = robarma::bip::rho2(e).sum();
            residuals[0] = est;
            return true;
        };
    };

    inline arma_fit bmm(const arma_model &model, const double &sigma, arma_fit &initial)
    {
        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<cost, 4>(new cost(model, sigma));

        ceres::Solver::Options options;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::bmm, cost_function, options);

        return fit;
    }

} // namespace robarma::bmm
// end of file
