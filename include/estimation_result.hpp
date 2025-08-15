
/**
 * @file estimation_result.hpp
 * @brief Estimation result types and enums for ARMA model fitting in robarma.
 *
 * Defines the estimation_method enum and the estimation_result class, which
 * encapsulate the outcome of an ARMA parameter estimation, including method,
 * convergence status, cost, and optional optimizer report.
 *
 * Used throughout the library to standardize reporting and tracking of estimation
 * results for both initial and final fits.
 *
 */
#pragma once

#include <iostream>
#include <iomanip>
#include <array>

namespace robarma
{
    /**
     * @brief Supported estimation methods for ARMA models.
     *
     * Used to tag results and select algorithms.
     */
    enum class estimation_method
    {
        hannan_rissanen,
        ols,
        mle,
        ftau,
        s,
        bs,
        mm,
        bmm,
        count // Helper to get the number of methods
    };

    inline const char *to_string(estimation_method method)
    {
        static constexpr std::array<const char *, static_cast<size_t>(estimation_method::count)> names{
            "Hannan-Rissanen", "OLS", "MLE", "FTAU", "S", "BS", "MM", "BMM"};
        size_t idx = static_cast<size_t>(method);
        if (idx < names.size())
            return names[idx];
        return "unknown";
    }

    inline std::ostream &operator<<(std::ostream &os, estimation_method method)
    {
        return os << to_string(method);
    }

    /**
     * @brief Stores the outcome of an ARMA parameter estimation.
     *
     * Contains:
     *  - method: which estimation method was used
     *  - convergence: whether the optimizer converged
     *  - final_cost: objective function value (as we return only one value, Ceres gives the correct value, not squared one)
     *  - report: (optional) optimizer report string
     *
     * Used in arma_fit to track both initial and final estimation results.
     */
    class estimation_result
    {
    public:
        estimation_method method;
        bool convergence;
        double final_cost;
        std::string report;

        estimation_result() {}

        // Used in closed-form solutions, ie. Hannan-Rissanen, no Ceres report as nothing is minimized.
        estimation_result(estimation_method method, bool convergence, double final_cost) : method{method}, convergence{convergence}, final_cost{final_cost}
        {
        }

        estimation_result(estimation_method method, bool convergence, double final_cost, std::string report) : method{method}, convergence{convergence}, final_cost{final_cost}, report{report}
        {
        }

        friend std::ostream &operator<<(std::ostream &os, const estimation_result &params)
        {
            auto format_number = [](double val)
            {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(4) << std::setw(18) << std::left << val;
                return ss.str();
            };

            auto format_bool = [](bool val)
            {
                std::string bool_string;
                bool_string = (val) ? "TRUE" : "FALSE";

                std::ostringstream ss;
                ss << std::fixed << std::setw(18) << std::left << bool_string;
                return ss.str();
            };

            os << std::left
               << std::setw(20) << "estimation method";
            os << std::fixed << std::setw(18) << std::left << params.method << " ";
            os << "\n"
               << std::left
               << std::setw(20) << "convergence";
            os << format_bool(params.convergence) << " ";
            os << "\n"
               << std::left
               << std::setw(20) << "final cost";
            os << format_number(params.final_cost) << "\n";
            return os;
        };
    };

}

// end of file