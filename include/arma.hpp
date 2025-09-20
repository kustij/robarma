
/**
 * @file arma.hpp
 * @brief Core ARMA model, parameter, and fit/result types for the RobARMA library.
 *
 * This header defines the main data structures for ARMA modeling and estimation:
 *  - arma_model: Holds the time series, model order, and basic statistics.
 *  - arma_params: Stores ARMA parameters (phi, theta, mu).
 *  - arma_fit: Stores the result of an estimation, including final and initial values.
 *
 * All classes are designed for use with robust and classical ARMA estimation methods,
 * and are compatible with Ceres optimization and Eigen for linear algebra.
 *
 */
#pragma once

#include <Eigen/Dense>
#include <iomanip>
#include <alias.hpp>
#include <bip.hpp>
#include <estimation_result.hpp>
#include <robust.hpp>
#include <optional>

namespace robarma
{

    /**
     * @brief Represents an ARMA model and its associated time series data.
     *
     * Holds the observed time series, model order (p, q), and basic statistics (mu, sigma).
     * Provides methods for extracting parameters and computing residuals.
     */
    class arma_model
    {
    public:
        Eigen::VectorXd y;
        int p;
        int q;
        int n;
        int r;
        int num_params;
        double sigma;
        double mu;

        arma_model(Eigen::VectorXd y, int p, int q) : y{y}, p{p}, q{q}
        {
            n = y.size();
            r = fmax(p, q);
            mu = robarma::base::median(y);
            sigma = robarma::base::scale<double>(y.array() - mu);
        }

        /**
         * @brief Unpack the ARMA params inside Ceres cost functions for optimization usage
         *
         * @tparam T
         * @param parameters
         * @return std::tuple<Vec<T>, Vec<T>, T>
         */
        template <typename T>
        std::tuple<Vec<T>, Vec<T>, T> get_params(T const *const *parameters) const
        {
            Vec<T> phi = Eigen::Map<const Vec<T>>(parameters[0], p);
            Vec<T> theta = Eigen::Map<const Vec<T>>(parameters[1], q);
            T mu = *parameters[2];

            return std::make_tuple(phi, theta, mu);
        }

        template <typename T>
        Vec<T> arma_residuals(Vec<T> phi, Vec<T> theta, T mu) const
        {
            Vec<T> e = Vec<T>::Zero(n);

            T ar;
            T ma;

            for (int i = r; i < n; i++)
            {
                ar = phi.dot(y.segment(i - p, p).reverse().template cast<T>());
                ma = theta.dot(e.segment(i - q, q).reverse().template cast<T>());
                e(i) = T(y(i)) - mu * (T(1) - phi.sum()) - ar - ma;
            }
            return e;
        }

        template <typename T>
        Vec<T> bip_arma_residuals(Vec<T> phi, Vec<T> theta, T mu, T sigma) const
        {
            Vec<T> e = Vec<T>::Zero(n);

            T ar;
            T rq;
            T rp;

            Vec<T> _rq;
            Vec<T> _rp;

            for (int i = r; i < n; i++)
            {
                ar = phi.dot(y.segment(i - p, p).reverse().template cast<T>() - e.segment(i - p, p).reverse());

                _rq = e.segment(i - q, q).reverse().array() / sigma;
                _rp = e.segment(i - p, p).reverse().array() / sigma;

                rq = theta.dot((sigma * bip::eta(_rq).array()).matrix());
                rp = phi.dot((sigma * bip::eta(_rp).array()).matrix());

                e(i) = T(y(i)) - mu * (T(1) - phi.sum()) - ar - rq - rp;
            }
            return e;
        }
    };

    /**
     * @brief Stores ARMA model parameters (phi, theta, mu).
     *
     * phi: AR coefficients (length p)
     * theta: MA coefficients (length q)
     * mu: mean parameter
     *
     * Can be constructed from Eigen vectors or raw pointers (for optimizer interop).
     */
    class arma_params
    {
    public:
        Eigen::VectorXd phi;
        Eigen::VectorXd theta;
        double mu;

        arma_params() = default;
        arma_params(const Eigen::VectorXd &phi, const Eigen::VectorXd &theta, double mu)
            : phi(phi), theta(theta), mu(mu) {}
        arma_params(double *_phi, int p, double *_theta, int q, double *_mu)
            : phi(Eigen::Map<Eigen::VectorXd>(_phi, p)),
              theta(Eigen::Map<Eigen::VectorXd>(_theta, q)),
              mu(*_mu) {}
    };

    /**
     * @brief Stores the result of ARMA parameter estimation.
     *
     * Contains:
     *  - model: reference to the ARMA model
     *  - params: estimated parameters (final)
     *  - result: estimation result (final)
     *  - initial_params: (optional) initial parameters used for optimization
     *  - initial_result: (optional) initial estimation result
     *
     * Used to track both the initial and final state of an estimation process.
     */
    struct arma_fit
    {
    public:
        const arma_model &model;
        arma_params params;
        estimation_result result;
        std::optional<arma_params> initial_params;
        std::optional<estimation_result> initial_result;

        arma_fit(const arma_model &model, const arma_params &params, estimation_result result,
                 std::optional<arma_params> initial_params = std::nullopt,
                 std::optional<estimation_result> initial_result = std::nullopt)
            : model{model}, params(params), result(result),
              initial_params(initial_params), initial_result(initial_result) {}

        friend std::ostream &operator<<(std::ostream &os, const arma_fit &params)
        {
            auto format_number = [](double val)
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(4) << std::setw(8) << std::right << val;
                return ss.str();
            };

            os << "ARMA estimation summary\n\n";

            if (params.initial_params)
            {
                os << "Initial values\n\n"
                   << std::left
                   << std::setw(8) << "phi";
                for (int i = 0; i < params.initial_params->phi.size(); ++i)
                    os << format_number(params.initial_params->phi[i]) << " ";
                os << "\n"
                   << std::setw(8) << "theta";
                for (int i = 0; i < params.initial_params->theta.size(); ++i)
                    os << format_number(params.initial_params->theta[i]) << " ";
                os << "\n"
                   << std::setw(8) << "mu"
                   << format_number(params.initial_params->mu) << "\n\n";
            }

            os << params.result << "\n";
            os << "Estimated parameters\n\n"
               << std::left
               << std::setw(8) << "phi";
            for (int i = 0; i < params.params.phi.size(); ++i)
                os << format_number(params.params.phi[i]) << " ";
            os << "\n"
               << std::setw(8) << "theta";
            for (int i = 0; i < params.params.theta.size(); ++i)
                os << format_number(params.params.theta[i]) << " ";
            os << "\n"
               << std::setw(8) << "mu"
               << format_number(params.params.mu) << "\n";

            return os;
        };
    };
}
// end of file