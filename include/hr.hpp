#pragma once

#include <Eigen/Dense>
#include <arma.hpp>
#include <estimation_result.hpp>

/**
 * @brief Initial estimators for estimating ARMA(p, q)-processes
 *
 */
namespace robarma::initial
{
    /**
     * @brief Hannan-Rissanen estimator
     * Fit an ARMA(p, q) process using Hannan-Rissanen estimator.
     * See \cite HannanRissanen
     * @param model
     * @return arma_fit
     */
    inline arma_fit hannan_rissanen(arma_model model)
    {
        // Step 1: Fit an AR(M)-model to data
        double mu = model.y.mean();

        int m = std::fmax(2 * model.p + 1, 2 * model.q + 1);

        // Build special case of Toeplitz matrix
        Eigen::MatrixXd ax = Eigen::MatrixXd::Zero(model.n - m, m);
        for (int i = 0; i < m; i++)
        {
            ax.col(i) << model.y.segment(m - i - 1, model.n - m).array() - mu;
        }

        Eigen::VectorXd yy = model.y.segment(m, model.n - m).array() - mu;
        Eigen::VectorXd ar = ax.householderQr().solve(yy);

        // Step 2: Fit a linear model
        Eigen::VectorXd ee = yy - ax * ar;

        int rr = std::fmax(model.p + 1, model.q + 1);
        int t = ee.size();

        Eigen::MatrixXd ay = Eigen::MatrixXd::Zero(t - rr, model.p);
        for (int i = 0; i < model.p; i++)
        {
            ay.col(i) << yy.segment(rr - i - 1, t - rr);
        }

        Eigen::MatrixXd ae = Eigen::MatrixXd::Zero(t - rr, model.q);
        for (int i = 0; i < model.q; i++)
        {
            ae.col(i) << ee.segment(rr - i - 1, t - rr);
        }

        Eigen::MatrixXd C(ay.rows(), ay.cols() + ae.cols());
        C << ay, ae;

        Eigen::VectorXd beta = C.householderQr().solve(yy.segment(rr, t - rr));

        Eigen::VectorXd phi = beta.segment(0, model.p);
        Eigen::VectorXd theta = beta.segment(model.p, model.q);

        estimation_result result = estimation_result(estimation_method::hannan_rissanen, true, 0.0);

        return arma_fit(model, arma_params(phi, theta, mu), result);
    }
} // namespace robarma::initial
