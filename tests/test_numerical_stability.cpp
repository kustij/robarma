#include <Eigen/Dense>
#include <arma.hpp>
#include <catch2/catch_test_macros.hpp>
#include <estimators.hpp>
#include <iostream>
#include <simulate.hpp>

TEST_CASE("MLE stability of ARMA(1,1) estimators", "[numerical-stability]")
{
    const int N = 1000;
    int mle_success = 0;

    Eigen::VectorXd phi(1);
    phi << 0.8;
    Eigen::VectorXd theta(1);
    theta << -0.7;
    double mu = 0;
    int T = 500;

    for (int i = 0; i < N; ++i)
    {
        Eigen::VectorXd innovations = robarma::generate_innovations_with_outliers(T, 0.1, 5);
        Eigen::VectorXd y = robarma::simulate(phi, theta, mu, T, innovations); // new innovations each time
        robarma::arma_model model(y, 1, 1);

        auto mle_fit = robarma::estimators::mle(model);
        // auto ftau_fit = robarma::estimators::ftau(model);

        if (mle_fit.result.convergence && mle_fit.initial_result.has_value() && mle_fit.initial_result->convergence)
            ++mle_success;
        // if (ftau_fit.result.convergence && ftau_fit.initial_result.has_value() && ftau_fit.initial_result->convergence)
        //     ++ftau_success;
    }

    double mle_pct = 100.0 * mle_success / N;
    // double ftau_pct = 100.0 * ftau_success / N;

    std::cout << "MLE convergence: " << mle_pct << "%\n";
    // std::cout << "FTAU convergence: " << ftau_pct << "%\n";

    // Optionally, require a minimum convergence rate
    REQUIRE(mle_pct > 80.0);
    // REQUIRE(ftau_pct > 80.0);
}

TEST_CASE("FTAU stability of ARMA(1,1) estimators", "[numerical-stability]")
{
    const int N = 1000;
    int ftau_success = 0;

    Eigen::VectorXd phi(1);
    phi << 0.8;
    Eigen::VectorXd theta(1);
    theta << -0.7;
    double mu = 0;
    int T = 500;

    for (int i = 0; i < N; ++i)
    {
        Eigen::VectorXd innovations = robarma::generate_innovations_with_outliers(T, 0.1, 5);
        Eigen::VectorXd y = robarma::simulate(phi, theta, mu, T, innovations); // new innovations each time
        robarma::arma_model model(y, 1, 1);

        auto ftau_fit = robarma::estimators::ftau(model);

        if (ftau_fit.result.convergence && ftau_fit.initial_result.has_value() && ftau_fit.initial_result->convergence)
            ++ftau_success;
    }

    double ftau_pct = 100.0 * ftau_success / N;

    std::cout << "FTAU convergence: " << ftau_pct << "%\n";

    // Optionally, require a minimum convergence rate
    REQUIRE(ftau_pct > 80.0);
}
