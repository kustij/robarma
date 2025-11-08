#pragma once

#include <arma.hpp>
#include <hr.hpp>
#include <solver.hpp>

#include <bip_s.hpp>
#include <bmm.hpp>
#include <estimation_result.hpp>
#include <ftau.hpp>
#include <mle.hpp>
#include <mm.hpp>
#include <ols.hpp>
#include <s.hpp>

namespace robarma::estimators
{
    inline double sigma_mle(const arma_fit &fit)
    {
        auto cost_obj = mle::cost(fit.model);
        auto [phi, theta, mu] = solver::get_pointers(fit);
        const double *const parameter_blocks[] = {phi, theta, mu};
        auto [f, v, w] = cost_obj.filter(parameter_blocks);
        return (v.array().square() / f.array()).mean();
    }

    inline double sigma_ols(const arma_fit &fit)
    {
        return fit.model.arma_residuals<double>(fit.params).array().square().sum() / fit.model.n;
    }

}