#pragma once

#include <arma.hpp>
#include <estimation_result.hpp>

#include <logging.hpp>

namespace robarma::solver
{
    /**
     * @brief Get safe pointers to parameter memory for Ceres optimization.
     *
     * Returns valid, non-overlapping pointers for phi, theta, and mu, even if phi/theta are empty.
     * This prevents pointer aliasing and undefined behavior when passing parameter blocks to Ceres.
     *
     * @param params arma_fit containing parameter vectors and mu
     * @return tuple of (phi_ptr, theta_ptr, mu_ptr)
     */
    inline std::tuple<double *, double *, double *> get_pointers(const arma_fit &params) noexcept
    {
        static double empty[] = {};
        double *phi_ptr = params.params.phi.size() ? const_cast<double *>(params.params.phi.data()) : empty;
        double *theta_ptr = params.params.theta.size() ? const_cast<double *>(params.params.theta.data()) : empty;
        double *mu_ptr = const_cast<double *>(&params.params.mu);
        return {phi_ptr, theta_ptr, mu_ptr};
    }

    /**
     * @brief Solve ARMA parameter estimation problem using Ceres optimizer.
     *
     * @param model The ARMA model structure (const ref)
     * @param initial The initial fit (const ref)
     * @param method The estimation method
     * @param cost_function The Ceres cost function (non-const pointer, as Ceres may mutate it)
     * @param options The Ceres solver options (const ref)
     * @return arma_fit containing the optimized parameters and results
     */
    template <typename T>
    arma_fit solve(const arma_model &model, const arma_fit initial, estimation_method method, ceres::DynamicAutoDiffCostFunction<T> *cost_function, ceres::Solver::Options options)
    {
        robarma::disable_ceres_logging();
        arma_fit opt_params = initial;

        auto [phi, theta, mu] = get_pointers(opt_params);

        ceres::Problem problem;

        cost_function->AddParameterBlock(model.p);
        cost_function->AddParameterBlock(model.q);
        cost_function->AddParameterBlock(1);
        cost_function->SetNumResiduals(1);

        problem.AddResidualBlock(cost_function, nullptr, phi, theta, mu);

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Use own success type instead of summary.IsSolutionUsable()
        // Successful only when convergence is reached
        bool success = (summary.termination_type == ceres::TerminationType::CONVERGENCE) ? true : false;

        // Evaluate the cost function value
        double cost = 0.0;
        const double *const parameter_blocks[] = {phi, theta, mu};

        cost_function->Evaluate(parameter_blocks, &cost, nullptr);

        estimation_result result = estimation_result(method, success, cost, summary.FullReport());
        arma_params params(phi, model.p, theta, model.q, mu);

        arma_fit fit(model, params, result, initial.params, initial.result);
        return fit;
    }
}

// end of file