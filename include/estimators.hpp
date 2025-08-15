

/**
 * @file estimators.hpp
 * @brief High-level ARMA(p, q)-estimators
 *
 * Provides entry points for fitting ARMA models using various estimation methods:
 *  - OLS, MLE, FTAU, S, MM, BIP-MM, BIP-S, etc.
 *
 * Each estimator returns an arma_fit object, encapsulating the model, parameters, and results.
 * These functions orchestrate the use of initial estimators and Ceres optimization.
 *
 */
#pragma once

#include <arma.hpp>
#include <hr.hpp>
#include <solver.hpp>

#include <ols.hpp>
#include <mle.hpp>
#include <ftau.hpp>
#include <s.hpp>
#include <mm.hpp>
#include <bmm.hpp>
#include <bip_s.hpp>
#include <estimation_result.hpp>

/**
 * @namespace robarma::estimators
 * @brief High-level ARMA(p, q)-estimators
 *
 * This namespace is user interface for fitting ARMA(p, q) processes.
 * Each estimator function returns an arma_fit object, and internally manages initial estimation, cost function setup, and optimization.
 */
namespace robarma::estimators
{
    /**
     * @brief Ordinary least squares estimator
     *
     * Fit an ARMA(p, q) process using ordinary least squares estimator.
     *
     * @param model
     * @return arma_fit
     */
    arma_fit ols(const arma_model &model)
    {
        arma_fit initial = robarma::initial::hannan_rissanen(model);

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<ols::cost, 4>(new ols::cost(model));

        // With trust-region minimizer, every solution is equal to initial estimate of Hannan-Rissanen.
        ceres::Solver::Options options;
        options.minimizer_type = ceres::LINE_SEARCH;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::ols, cost_function, options);

        return fit;
    }

    /**
     * @brief Maximum likelihood estimator
     *
     * Fit an ARMA(p, q) process using maximum likelihood estimator.
     * See \cite HarveyPhillips1979
     * @param model
     * @return arma_fit
     */
    arma_fit mle(const arma_model &model)
    {
        arma_fit initial = robarma::initial::hannan_rissanen(model);

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<mle::cost, 4>(new mle::cost(model));

        ceres::Solver::Options options;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::mle, cost_function, options);

        return fit;
    }

    /**
     * @brief Filtered tau-estimator
     *
     * Fit an ARMA(p, q) process using filtered tau-estimator.
     * See \cite Bianco

     * @param model
     * @return arma_fit
     */
    arma_fit ftau(const arma_model &model)
    {
        arma_fit initial = robarma::initial::hannan_rissanen(model);

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<ftau::cost, 4>(new ftau::cost(model));

        ceres::Solver::Options options;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::ftau, cost_function, options);

        return fit;
    }

    /**
     * @brief S-estimator
     *
     * Fit an ARMA(p, q) process using S-estimator.
     * Definition and rho-functions are as shown in \cite Muler
     * @param model
     * @return arma_fit
     */
    arma_fit s(const arma_model &model)
    {
        arma_fit initial = robarma::initial::hannan_rissanen(model);

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<s::cost, 4>(new s::cost(model));

        // Unstable without line_search
        ceres::Solver::Options options;
        options.minimizer_type = ceres::LINE_SEARCH;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::s, cost_function, options);

        return fit;
    }

    /**
     * @brief MM-estimator
     *
     * Fit an ARMA(p, q) process using filtered MM-estimator.
     * Definition and rho-functions are as shown in \cite Muler
     * @param model
     * @return arma_fit
     */
    arma_fit mm(const arma_model &model)
    {
        arma_fit initial = robarma::estimators::s(model);

        double sigma = initial.result.final_cost;

        auto *cost_function = new ceres::DynamicAutoDiffCostFunction<mm::cost, 4>(new mm::cost(model, sigma));

        ceres::Solver::Options options;

        arma_fit fit = robarma::solver::solve(model, initial, estimation_method::mm, cost_function, options);

        return fit;
    }

    /**
     * @brief BIP-MM-estimator
     *
     * Fit an ARMA(p, q) process using filtered BIP-MM-estimator.
     * Definition and rho-functions are as shown in \cite Muler
     * @param model
     * @return arma_fit
     */
    arma_fit bip_mm(const arma_model &model)
    {
        // Step 1.
        arma_fit s_mm = robarma::estimators::s(model);
        arma_fit s_bmm = robarma::estimators::bip_s(model);

        // Step 2.
        double sigma = fmin(s_mm.result.final_cost, s_bmm.result.final_cost);

        // Step 3.
        arma_fit fit_mm = robarma::mm::mm(model, sigma, s_mm);
        arma_fit fit_bmm = robarma::bmm::bmm(model, sigma, s_bmm);

        double m = fit_mm.result.final_cost;
        double mb = fit_bmm.result.final_cost;

        return (m < mb) ? fit_mm : fit_bmm;
    }
}

// end of file