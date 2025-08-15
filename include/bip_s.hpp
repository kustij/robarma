#pragma once

#include <arma.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <hr.hpp>
#include <alias.hpp>
#include <bip.hpp>
#include <robust.hpp>
#include <ts.hpp>

namespace robarma::estimators
{
    struct bip_s_functor
    {
    private:
        arma_model model;

    public:
        bip_s_functor(const arma_model &model)
            : model(model)
        {
        }

        template <typename T>
        T bip_sigma(Vec<T> phi, Vec<T> theta) const
        {
            // Techrep p. 7, (19)
            T kappa = T(0.8725);
            return model.sigma / (T(1) + ceres::pow(kappa, 2) * causal(phi, theta).array().square().sum());
        }

        template <typename T>
        bool operator()(T const *const *parameters, T *residuals) const
        {
            auto [phi, theta, mu] = model.get_params(parameters);
            // Calculate scale with rho1
            // delta is b = a/2, and a = max rho1
            T sigma = bip_sigma(phi, theta);

            T delta = T(3.25 / 2);
            std::function<Vec<T>(Vec<T>)> func = static_cast<Vec<T> (*)(const Vec<T>)>(&robarma::bip::rho1);
            T est = robarma::base::scale(model.bip_arma_residuals(phi, theta, mu, sigma), delta, func);
            residuals[0] = est;
            return true;
        };
    };

    arma_fit bip_s(const arma_model &model)
    {
        // Calculate the initial S-estimator for ARMA model
        arma_fit initial = robarma::initial::hannan_rissanen(model);

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<bip_s_functor, 4>(new bip_s_functor(model));

        ceres::Solver::Options options;
        options.minimizer_type = ceres::LINE_SEARCH;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::bs, cost_function, options);
        return fit;
    }
}
// end of file