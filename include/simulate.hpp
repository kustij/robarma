/**
 * @file simulate.hpp
 * @brief Simulation of ARMA(p, q) process with normal errors
 *
 */

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>
#include <EigenRand/EigenRand>

namespace robarma
{

    bool stationary(const Eigen::VectorXd &ar)
    {
        Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
        Eigen::VectorXd coeff = Eigen::VectorXd(ar.size() + 1);
        coeff << Eigen::VectorXd::Ones(1), -ar;
        solver.compute(coeff);
        Eigen::VectorXcd roots = solver.roots();
        return (roots.array().abs() > 1.0).all();
    }

    bool invertible(const Eigen::VectorXd &ma)
    {
        Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
        Eigen::VectorXd coeff = Eigen::VectorXd(ma.size() + 1);
        coeff << Eigen::VectorXd::Ones(1), ma;
        solver.compute(coeff);
        Eigen::VectorXcd roots = solver.roots();
        return (roots.array().abs() > 1.0).all();
    }

    /**
     * @brief Simulate an ARMA(p, q) process
     *
     * @param ar AR parameters
     * @param ma MA parameters
     * @param mu location parameter
     * @param n sample size
     * @param burn_in size of burn in period
     * @param seed random seed
     * @return Eigen::VectorXd
     */
    /**
     * @brief Simulate an ARMA(p, q) process
     *
     * @param phi AR parameters (optional, default: none)
     * @param theta MA parameters (optional, default: none)
     * @param mu location parameter
     * @param n sample size
     * @param burn_in size of burn in period (default: 100)
     * @param seed random seed (default: 0, uses current time)
     * @return Eigen::VectorXd
     */
    inline Eigen::VectorXd simulate(
        const Eigen::VectorXd &phi = Eigen::VectorXd{},
        const Eigen::VectorXd &theta = Eigen::VectorXd{},
        double mu = 0.0,
        int n = 100,
        int burn_in = 100,
        int seed = 0)
    {
        if (seed == 0)
            seed = static_cast<int>(std::time(nullptr));

        int nn = burn_in + n;
        int p = phi.size();
        int q = theta.size();
        int r = std::max(p, q);

        if (p > 0 && !stationary(phi))
            throw std::invalid_argument("AR parameters must specify a stationary process.");

        if (q > 0 && !invertible(theta))
            throw std::invalid_argument("MA parameters must specify an invertible process.");

        Eigen::Rand::Vmt19937_64 urng{static_cast<unsigned long long>(seed)};

        Eigen::VectorXd e = Eigen::Rand::normal<Eigen::VectorXd>(nn, 1, urng);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(nn);

        Eigen::VectorXd phi_tmp;
        Eigen::VectorXd theta_tmp;

        if (p > 0 && q == 0)
        {
            for (int i = r + 1; i < nn; i++)
            {
                phi_tmp = x.segment(i - p, p).reverse();
                x(i) = mu * (1.0 - phi.sum()) + e(i) + phi.dot(phi_tmp);
            }
        }
        else if (p == 0 && q > 0)
        {
            for (int i = r + 1; i < nn; i++)
            {
                theta_tmp = e.segment(i - q, q).reverse();
                x(i) = mu + e(i) + theta.dot(theta_tmp);
            }
        }
        else if (p > 0 && q > 0)
        {
            for (int i = r + 1; i < nn; i++)
            {
                phi_tmp = x.segment(i - p, p).reverse();
                theta_tmp = e.segment(i - q, q).reverse();
                x(i) = mu * (1.0 - phi.sum()) + e(i) + phi.dot(phi_tmp) + theta.dot(theta_tmp);
            }
        }
        return x.tail(n);
    }
}

// end of file