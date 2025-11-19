# RobARMA - Robust estimators for ARMA models

RobARMA is a small C++ library of robust and classic estimators for ARMA(p, q) processes. To achieve fast estimation of ARMA models, RobARMA builds on top of Eigen and Ceres.

## Features

- Robust and classic estimators for ARMA(p, q) processes
- Simulation of ARMA(p, q) processes

## Estimators

- Robust estimators:
  - S
  - FTAU (filtered tau)
  - MM
  - BIP-MM (bounded innovation propagation MM)
- Classic estimators:
  - OLS (ordinary least squares)
  - MLE (maximum likelihood via Kalman filter)

## General

ARMA(p, q)-process $y_t$ is assumed as stationary and invertible and of form

$$
\phi(B)(y_t - \mu) = \theta(B)\varepsilon_t,
$$

where $\varepsilon_t$ are i.i.d. random variables with symmetric distribution and $\phi(B)$ and $\theta(B)$ are characteristic polynomials given by

$$
\phi(B) = 1 - \sum_{i = 1}^p \phi_i B^i \\
\theta(B) = 1 + \sum_{i = 1}^q \theta_i B^i.
$$

Hence, $y_t$ can be expressed as

$$
y_t = \varepsilon_t + \mu(1 - \sum_{i = 1}^p\phi_i) + \sum_{i = 1}^p\phi_i y_{t - i} + \sum_{i = 1}^q \theta_i \varepsilon_{t-i} .
$$

## Quickstart

Estimators can be found from namespace `robarma::estimators`

```cpp
#include <robarma/robarma.hpp>

// Simulate a MA(2) process of 100000 observations.

Eigen::VectorXd theta = Eigen::VectorXd::Zero(2);

theta << 0.2, -0.4;
double mu = 2;

Eigen::VectorXd y = robarma::simulate({}, theta, mu, 10000);

// Create an ARMA model of the simulated time series.

robarma::arma arma(y, 0, 2);

// Fit a MM-estimator and print the estimation result

arma_fit fit = robarma::estimators::mm(arma);
std::cout << fit << std::endl;

```

## Build

## Building RobARMA

RobARMA uses CMake for building and vcpkg for dependency management. You can also build without vcpkg if you manually install dependencies (Eigen3, Ceres).

### Recommended: Build with vcpkg

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

Replace `/path/to/vcpkg` with your vcpkg installation path.

### Manual build (without vcpkg)

1. Install Eigen3, Ceres, and Catch2 using your system package manager or from source.

2. Configure and build:
   ```bash
   cmake -B build -S .
   cmake --build build
   ```

### Notes

- By default, tests are not built. To enable tests (requires Catch2):
  ```bash
  cmake -B build -S . -DROBARMA_BUILD_TESTS=ON
  ```
- The library is header-only. You can use the headers directly if you manually set include paths and dependencies.

### Logging Suppression (Ceres/glog)

RobARMA itself doesn't log anything. However, Ceres logs heavily using glog and directs this to console. By default, RobARMA suppresses all logging from Ceres Solver (and glog). This is done automatically when you use any estimator or solver function.

If you want to see Ceres/glog logs, you can opt out by defining the macro `ROBARMA_ENABLE_CERES_LOGGING` in your build:

**CMake:**

```cmake
target_compile_definitions(robarma INTERFACE ROBARMA_ENABLE_CERES_LOGGING)
```

**Manual build:**

```
-DROBARMA_ENABLE_CERES_LOGGING
```

If you use your own glog setup, initialize glog and set flags before calling RobARMA functions.
