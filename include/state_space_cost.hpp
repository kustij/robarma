#pragma once

#include <alias.hpp>
#include <unsupported/Eigen/KroneckerProduct>

namespace robarma
{

    struct state_space_cost
    {
    protected:
        arma_model model;
        int r;

    public:
        state_space_cost(arma_model model)
            : model(model)
        {
            r = fmax(model.p, model.q + 1);
        }

        template <typename T>
        Mat<T> F0(const Vec<T> phi) const
        {
            Mat<T> F = Mat<T>::Zero(r, r);
            F.topRightCorner(r - 1, r - 1).setIdentity();
            F.col(0).segment(0, model.p) = phi;
            return F;
        }

        template <typename T>
        Vec<T> H0(const Vec<T> theta) const
        {
            Vec<T> H = Vec<T>::Zero(r);
            H.head(1).setOnes();
            H.segment(1, model.q) = theta;
            return H;
        }

        template <typename T>
        Mat<T> P0(const Mat<T> F, const Vec<T> H) const
        {
            Mat<T> S = Mat<T>::Identity(pow(r, 2), pow(r, 2)) - kroneckerProduct(F, F);
            Mat<T> V = H * H.transpose();
            V.resize(r * r, 1);
            Mat<T> P0 = S.householderQr().solve(V);
            P0.resize(r, r);
            return P0;
        }

        template <typename T>
        Vec<T> z0() const
        {
            Vec<T> z = Vec<T>::Zero(r);
            z.head(1).setOnes();
            return z;
        }

        template <typename T>
        Vec<T> c0(const Vec<T> phi, const T mu) const
        {
            Vec<T> c = Vec<T>::Zero(r);
            c(0) = mu * (T(1) - phi.sum());
            return c;
        }
    };

} // namespace robarma

// end of file
