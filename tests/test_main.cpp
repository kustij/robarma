#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <simulate.hpp>
#include <Eigen/Dense>
#include <arma.hpp>
#include <estimators.hpp>
#include <mle.hpp>
#include <ftau.hpp>
#include <s.hpp>
#include <mm.hpp>
#include <bip_s.hpp>
#include <tau.hpp>
#include <ceres/ceres.h>
#include <ts.hpp>
#include <robust.hpp>

TEST_CASE("ARMA TEST", "[arma]")
{

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    theta << 0.2, -0.4;
    double mu = 2;

    Eigen::VectorXd y = robarma::simulate({}, theta, mu, 10000);

    // Create an ARMA model of the simulated time series.

    robarma::arma_model arma(y, 0, 2);

    // Fit a MM-estimator and print the estimation result

    robarma::arma_fit fit = robarma::estimators::mm(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA HR - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 2, 10000);

    robarma::arma_model arma(y, 1, 2);

    std::cout << robarma::initial::hannan_rissanen(arma) << std::endl;
}

TEST_CASE("ARMA HR - 02", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(2);

    phi << -0.7, -0.5;

    Eigen::VectorXd y = robarma::simulate(phi, {}, 8, 1000);

    robarma::arma_model arma(y, 2, 0);
    std::cout << robarma::initial::hannan_rissanen(arma) << std::endl;
}

TEST_CASE("ARMA MM - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 0, 5000);

    robarma::arma_model arma(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::mm(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA S - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 0, 10000);

    robarma::arma_model arma(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::s(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA BIP-S - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 0, 10000);

    robarma::arma_model arma(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::bip_s(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA BIP-MM - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 0, 10000);

    robarma::arma_model arma(y, 1, 2);

    robarma::arma_fit fit = robarma::estimators::bip_mm(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA OLS - 1", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 2, 10000);

    robarma::arma_model model(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::ols(model);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA OLS - 2", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(2);

    phi << 0.7, -0.1;

    Eigen::VectorXd y = robarma::simulate(phi, {}, 2, 10000);

    robarma::arma_model model(y, 2, 0);
    robarma::arma_fit fit = robarma::estimators::ols(model);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA OLS - 3", "[fit]")
{
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    theta << -0.4, 0.8;

    Eigen::VectorXd y = robarma::simulate({}, theta, 3, 10000);

    robarma::arma_model arma(y, 0, 2);
    robarma::arma_fit fit = robarma::estimators::ols(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA MLE - 01", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 3, 10000);

    robarma::arma_model arma(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::mle(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA MLE - 02", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(3);

    phi << 0.7, -0.5, 0.4;

    Eigen::VectorXd y = robarma::simulate(phi, {}, 3, 10000);

    robarma::arma_model arma(y, 3, 0);
    robarma::arma_fit fit = robarma::estimators::mle(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA MLE - 03", "[arma]")
{
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    theta << -0.4, 0.8;

    Eigen::VectorXd y = robarma::simulate({}, theta, 3, 10000);

    robarma::arma_model arma(y, 0, 2);
    robarma::arma_fit fit = robarma::estimators::mle(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA MLE - 04", "[arma]")
{
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(3);

    theta << -0.5, 0.7, -0.2;

    Eigen::VectorXd y = robarma::simulate({}, theta, 1, 10000);

    robarma::arma_model arma(y, 0, 3);
    robarma::arma_fit fit = robarma::estimators::mle(arma);
    std::cout << fit << std::endl;
}

TEST_CASE("ARMA FTAU", "[arma]")
{
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

    phi << 0.7;
    theta << 0.2, -0.4;

    Eigen::VectorXd y = robarma::simulate(phi, theta, 2, 10000);

    robarma::arma_model arma(y, 1, 2);
    robarma::arma_fit fit = robarma::estimators::ftau(arma);
    std::cout << fit << std::endl;
}
