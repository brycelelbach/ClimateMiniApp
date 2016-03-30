/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_2B2ACBFF_7B1E_4A38_8957_B4FA065EC16A)
#define CHOMBO_2B2ACBFF_7B1E_4A38_8957_B4FA065EC16A

#include <array>

#include "Lapack.H"
#include "LapackWrapper.H"

#include "CMAConfiguration.H"
#include "CMAProblemState.H"

namespace climate_mini_app
{

struct advection_diffusion_profile 
{
    advection_diffusion_profile()
    // {{{
      : config()
      , cx(), cy(), cz()
      , vx(), vy(), kz()
    {} // }}}

    advection_diffusion_profile(
        configuration config_
      , Real cx_, Real cy_, Real cz_
      , Real vx_, Real vy_, Real kz_
        )
    // {{{
      : config(config_)
      , cx(cx_), cy(cy_), cz(cz_)
      , vx(vx_), vy(vy_), kz(kz_)
    {} // }}}
    
    // TODO: Implement caching in profile.
    void resetDt(Real)
    { // {{{
    } // }}}

    IntVect ghostVect() const
    { // {{{
        return IntVect(3, 3, 0);
    } // }}}

    std::array<Real, 3> face_coords(IntVect here) const
    { // {{{
        std::array<Real, 3> c;
        c[0] = Real(here[0])/(Real(config.nh));
        c[1] = Real(here[1])/(Real(config.nh));
        c[2] = Real(here[2])/(Real(config.nv));
        return c;
    } // }}}

    std::array<Real, 3> center_coords(IntVect here) const
    { // {{{
        std::array<Real, 3> c;
        c[0] = face_coords(here)[0]+0.5*dp()[0];
        c[1] = face_coords(here)[1]+0.5*dp()[1];
        c[2] = face_coords(here)[2]+0.5*dp()[2];
        return c;
    } // }}}

    // Spatial step size
    std::array<Real, 3> dp() const
    { // {{{
        return face_coords(IntVect(1, 1, 1));
    } // }}}

    // Time step size
    Real dt() const
    { // {{{
        // Check if advection is on. 
        if (!explicitOpDisabled())
        {
            // For advection: 
            //
            //      dt = CFL * dh / (vx + vy) 
            // 
            //      with CFL < 1.0 
            // 
            Real constexpr CFL = 0.8;
            Real const dh = dp()[0]; 

            return CFL * dh / (std::abs(vx) + std::abs(vy));
        }

        // Diffusion only.
        else
        {
            // We want something with the same magnitude as dv.
            Real constexpr A = 0.8;
            Real const dv = dp()[2];

            return A * dv;
        }
    } // }}}

    ProblemDomain problem_domain() const
    { // {{{
        IntVect lower_bound(IntVect::Zero);
        IntVect upper_bound(
            (config.nh - 1),
            (config.nh - 1),
            (config.nv - 1)
        );

        bool is_periodic[] = { true, true, false };
        ProblemDomain domain(lower_bound, upper_bound, is_periodic);

        return domain;
    } // }}} 

    bool implicitOpDisabled() const
    { // {{{ 
        // kz == 0
        return std::fabs(kz - 0.0) < 1e-14;
    } // }}}

    bool explicitOpDisabled() const
    { // {{{
        // (vx == 0) && (vy == 0)
        return (std::fabs(vx - 0.0) < 1e-14) && (std::fabs(vy - 0.0) < 1e-14);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////

    // FIXME: handle negative velocity
    template <std::size_t Dir, typename Integral>
    Real horizontal_flux_stencil(
        Real const* p
      , Integral idx
      , Integral step 
        ) const
    { // {{{
        static_assert((0 == Dir) || (1 == Dir)
                    , "Illegal Dir, should be 0 or 1");

        auto const v = (0 == Dir ? vx : vy);
        auto constexpr denom = 1.0 / 60.0;

        return -v * denom * ( 2.0  * p[idx - 3 * step]
                            - 13.0 * p[idx - 2 * step]
                            + 47.0 * p[idx -     step]
                            + 27.0 * p[idx           ]
                            - 3.0  * p[idx +     step]);
    } // }}}

    Real inverse_dh_c() const
    { // {{{
        Real const dh = dp()[0]; 

        return (1.0 / dh); 
    } // }}}

    template <std::size_t Dir, typename Integral>
    Real horizontal_stencil_1d(
        Real const* p
      , Integral idx
      , Integral step 
      , Real inv_dh_c
        ) const
    { // {{{
        static_assert((0 == Dir) || (1 == Dir)
                    , "Illegal Dir, should be 0 or 1");

        return inv_dh_c * (p[idx + step] - p[idx]);
    } // }}}

    template <std::size_t Stage, typename X, std::size_t TotalStages>
    void explicitOp(
        Real t
      , Real dt
      , Real const s_aE[TotalStages][TotalStages]
      , Real const s_b[TotalStages]
      , X& state_
      , X& kE_
      , std::array<X, TotalStages>& phi_
        ) const
    { // {{{
        if (explicitOpDisabled())
            return;

        // For HPX, we operate on problem_state_fabs (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        #if defined(CH_HPX)
            phi_[Stage].exchangeSync();
        #endif

        auto& state = state_.U();
        auto& phi   = phi_[Stage].U();
        auto& FX    = phi_[Stage].FX();
        auto& FY    = phi_[Stage].FY();

        Box const b = state_.box();

        IntVect const lower_phi = b.smallEnd(); // FIXME: Better names.
        IntVect const upper_phi = b.bigEnd();   // FIXME: Better names.
 
        IntVect const lower_FX  = surroundingNodes(b, 0).smallEnd();  // FIXME: Better names.
        IntVect const upper_FX  = surroundingNodes(b, 0).bigEnd();    // FIXME: Better names.

        IntVect const lower_FY  = surroundingNodes(b, 1).smallEnd();  // FIXME: Better names.
        IntVect const upper_FY  = surroundingNodes(b, 1).bigEnd();    // FIXME: Better names.

        IntVect phys_lower_phi = lower_phi;
        IntVect phys_upper_phi = upper_phi; 

        phys_lower_phi.shift(ghostVect());
        phys_upper_phi.shift(-1*ghostVect());

        IntVect phys_lower_FX = lower_FX;
        IntVect phys_upper_FX = upper_FX; 

        phys_lower_FX.shift(0, ghostVect()[0]);
        phys_upper_FX.shift(0, -2);
        phys_lower_FX.shift(1, ghostVect()[1]);
        phys_upper_FX.shift(1, -1*ghostVect()[1]);
        phys_lower_FX.shift(2, ghostVect()[2]);
        phys_upper_FX.shift(2, -1*ghostVect()[2]);

        IntVect phys_lower_FY = lower_FY;
        IntVect phys_upper_FY = upper_FY; 

        phys_lower_FY.shift(0, ghostVect()[0]);
        phys_upper_FY.shift(0, -1*ghostVect()[0]);
        phys_lower_FY.shift(1, ghostVect()[1]);
        phys_upper_FY.shift(1, -2);
        phys_lower_FY.shift(2, ghostVect()[2]);
        phys_upper_FY.shift(2, -1*ghostVect()[2]);

        ///////////////////////////////////////////////////////////////////////

        Real* pstate = state.dataPtr(0); 
        Real* pfx    = FX.dataPtr(0);
        Real* pfy    = FY.dataPtr(0);

        std::array<Real*, TotalStages> pphi;

        for (std::size_t s = 0; s < TotalStages; ++s)
            pphi[s] = phi_[s].U().dataPtr(0);

        Real const inv_dh_c = inverse_dh_c();

        // len0 is the extent of i and the stride of j.
        // len1 is the extent of j.
        // len0 * len1 is the stride of k.
        auto const len0_phi = (upper_phi[0] - lower_phi[0] + 1);
        auto const len1_phi = (upper_phi[1] - lower_phi[1] + 1);

        auto const len0_FX  = (upper_FX[0] - lower_FX[0] + 1);
        auto const len1_FX  = (upper_FX[1] - lower_FX[1] + 1);

        auto const len0_FY  = (upper_FY[0] - lower_FY[0] + 1);
        auto const len1_FY  = (upper_FY[1] - lower_FY[1] + 1);

        ///////////////////////////////////////////////////////////////////////
        // X flux computation

        // FIXME: Cleanup indexing.

        for (int k = phys_lower_FX[2]; k <= phys_upper_FX[2]; ++k)
        {
            auto const dif2_phi = k - lower_phi[2];
            auto const dif2_FX  = k - lower_FX[2];

            for (int j = phys_lower_FX[1]; j <= phys_upper_FX[1]; ++j)
            {
                auto const dif1_phi = j - lower_phi[1];
                auto const dif1_FX  = j - lower_FX[1];

                auto const idx_phi =
                    -lower_phi[0] + len0_phi * (dif1_phi + len1_phi * dif2_phi);
                auto const idx_FX =
                    -lower_FX[0] + len0_FX * (dif1_FX + len1_FX * dif2_FX);

                #pragma simd
                for (int i = phys_lower_FX[0]; i <= phys_upper_FX[0]; ++i)
                {
                    pfx[i + idx_FX] = horizontal_flux_stencil<0>
                        (pphi[Stage], i + idx_phi, 1);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Y flux computation

        // FIXME: Cleanup indexing.

        for (int k = phys_lower_FY[2]; k <= phys_upper_FY[2]; ++k)
        {
            auto const dif2_phi = k - lower_phi[2];
            auto const dif2_FY  = k - lower_FY[2];

            for (int j = phys_lower_FY[1]; j <= phys_upper_FY[1]; ++j)
            {
                auto const dif1_phi = j - lower_phi[1];
                auto const dif1_FY  = j - lower_FY[1];

                auto const idx_phi =
                    -lower_phi[0] + len0_phi * (dif1_phi + len1_phi * dif2_phi);
                auto const idx_FY =
                    -lower_FY[0] + len0_FY * (dif1_FY + len1_FY * dif2_FY);

                #pragma simd
                for (int i = phys_lower_FY[0]; i <= phys_upper_FY[0]; ++i)
                {
                    pfy[i + idx_FY] = horizontal_flux_stencil<1>
                        (pphi[Stage], i + idx_phi, len0_phi);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Advection computation
 
        for (int k = phys_lower_phi[2]; k <= phys_upper_phi[2]; ++k)
        {
            auto const dif2_phi = k - lower_phi[2];
            auto const dif2_FX  = k - lower_FX[2];
            auto const dif2_FY  = k - lower_FY[2];

            for (int j = phys_lower_phi[1]; j <= phys_upper_phi[1]; ++j)
            {
                auto const dif1_phi = j - lower_phi[1];
                auto const dif1_FX  = j - lower_FX[1];
                auto const dif1_FY  = j - lower_FY[1];

                auto const idx_phi =
                    -lower_phi[0] + len0_phi * (dif1_phi + len1_phi * dif2_phi);
                auto const idx_FX =
                    -lower_FX[0] + len0_FX * (dif1_FX + len1_FX * dif2_FX);
                auto const idx_FY =
                    -lower_FY[0] + len0_FY * (dif1_FY + len1_FY * dif2_FY);

                ///////////////////////////////////////////////////////////////
                // X component 

                #pragma simd
                for (int i = phys_lower_phi[0]; i <= phys_upper_phi[0]; ++i)
                {
                    Real phi_here = horizontal_stencil_1d<0>
                        (pfx, i + idx_FX, 1, inv_dh_c);

                    #pragma unroll
                    for (std::size_t s = Stage + 1; s < TotalStages; ++s)
                        pphi[s][i + idx_phi] += phi_here * dt * s_aE[s][Stage];

                    pstate[i + idx_phi] += phi_here * dt * s_b[Stage];
                }

                ///////////////////////////////////////////////////////////////
                // Y component 

                #pragma simd
                for (int i = phys_lower_phi[0]; i <= phys_upper_phi[0]; ++i)
                {
                    Real phi_here = horizontal_stencil_1d<1>
                        (pfy, i + idx_FY, len0_FY, inv_dh_c);

                    #pragma unroll
                    for (std::size_t s = Stage + 1; s < TotalStages; ++s)
                        pphi[s][i + idx_phi] += phi_here * dt * s_aE[s][Stage];

                    pstate[i + idx_phi] += phi_here * dt * s_b[Stage];
                }
            }
        }
    } // }}}

    ///////////////////////////////////////////////////////////////////////////

    void build_vertical_operator_for_solve(
        LapackFactorization& A
      , Box b
      , Real dtscale
        ) const
    { // {{{
        Real const c1 = 1.0;
        Real const c2 = -(kz * dt() * dtscale);
        build_vertical_operator(A, b, c1, c2);
    } // }}}

    void build_vertical_operator_for_apply(
        LapackFactorization& A
      , Box b
        ) const
    { // {{{
        Real const c1 = 0.0;
        Real const c2 = kz;
        build_vertical_operator(A, b, c1, c2);
    } // }}}

    void build_vertical_operator(
        LapackFactorization& A
      , Box b
      , Real c1
      , Real c2 
        ) const
    { // {{{
        std::uint64_t const size = b.size()[2];
        Real const dv = dp()[2]; 

        Real const coef = c2 / (1.0 * dv * dv); 
        Real const s[] = {0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0};

        int const KU = 3;
        int const KL = 3;

        A.setZero();
        
        int ncol = A.numCols();
        for (int col = 0; col < ncol; ++col) 
            for (int ix = -KU; ix <= KL; ++ix)
            {
                int row = col + ix;
                if ((row >= 0) && (row < ncol))
                    A(row, col) = ((row == col) ? c1 : 0) + s[ix + KU] * coef;
            }
        
        // Fix the diagonal ends for the homogeneous Neumann bcs
        A(0, 0) = c1 - 1.0 * coef;
        A(ncol - 1, ncol - 1) = c1 - 1.0 * coef;

/*
        // The gradient flux for the first interior face
        int const lenG = 4;
        Real const sG1[] = {-145.0, 159.0, -15.0, 1.0};
        Real const coefG1 = c2 / (120.0 * dv * dv);

        // The gradient flux for the second interior (regular) face
        Real const sG2[] = {1.0, -15.0, 15.0, -1.0};
        Real const coefG2 = c2 / (12.0 * dv * dv);

        for (int ix = 0; ix < lenG; ++ix)
        {
            A(0, ix) =
                ((ix == 0) ? c1 : 0) + sG1[ix] * coefG1;
            A(1, ix) =
                ((ix == 1) ? c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;

            int ixflip = ncol - ix - 1;
            A(ncol - 1, ixflip) =
                ((ix == 0) ? c1 : 0) + sG1[ix] * coefG1;
            A(ncol - 2, ixflip) =
                ((ix == 1) ? c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;
        }
*/
    } // }}}

    void define_vertical_operator(
        LapackFactorization& A
      , Box b
        ) const
    { // {{{
        std::uint64_t const size = b.size()[2];

        int const KU = 3;
        int const KL = 3;

        A.define(size, KL, KU);
    } // }}}

    template <std::size_t Stage, typename X, std::size_t TotalStages>
    void implicitOp(
        Real t
      , Real dt
      , Real const s_aI[TotalStages][TotalStages]
      , Real const s_b[TotalStages]
      , X& state_
      , X& kI_
      , std::array<X, TotalStages>& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
            return;

        auto& state = state_.U();
        auto& kI  = kI_.U();
        auto& phi = phi_[Stage].U();
 
        IntVect lower = state_.box().smallEnd();
        IntVect upper = state_.box().bigEnd(); 

        lower.shift(ghostVect());
        upper.shift(-1*ghostVect());

        Box b(lower, upper);

        std::uint64_t const size = b.size()[2];

        LapackFactorization A;
        define_vertical_operator(A, b);

        std::vector<Real> kI_buf(size), phi_buf(size);

        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int i = lower[0]; i <= upper[0]; ++i)
            {
                build_vertical_operator_for_apply(A, b);

                for (int k = 0; k <= upper[2] - lower[2]; ++k)
                    phi_buf[k] = phi(IntVect(i, j, k + lower[2]));

                LapackWrapper::applyBandMatrix(phi_buf.data(), kI_buf.data(), A);

                for (int k = 0; k <= upper[2] - lower[2]; ++k)
                {
                    #pragma unroll
                    for (std::size_t s = Stage + 1; s < TotalStages; ++s)
                        phi_[s].U()(IntVect(i, j, k + lower[2]))
                            += kI_buf[k] * dt * s_aI[s][Stage];

                    state(IntVect(i, j, k + lower[2]))
                        += kI_buf[k] * dt * s_b[Stage];
                }
            }
    } // }}}

    template <std::size_t Stage, typename X>
    void solve(
        Real t
      , Real dtscale
      , X&& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
            return;

        auto& phi = phi_.U();

        IntVect lower = phi_.box().smallEnd();
        IntVect upper = phi_.box().bigEnd(); 

        lower.shift(ghostVect());
        upper.shift(-1*ghostVect());

        Box b(lower, upper);

        std::uint64_t const size = b.size()[2];

        LapackFactorization A;
        define_vertical_operator(A, b);

        std::vector<Real> phi_buf(size);

        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int i = lower[0]; i <= upper[0]; ++i)
            {
                build_vertical_operator_for_solve(A, b, dtscale);

                for (int k = 0; k <= upper[2] - lower[2]; ++k)
                    phi_buf[k] = phi(IntVect(i, j, k + lower[2]));

                LapackWrapper::factorBandMatrix(A);
                LapackWrapper::solveBandMatrix(A, phi_buf.data());

                for (int k = 0; k <= upper[2] - lower[2]; ++k)
                    phi(IntVect(i, j, k + lower[2])) = phi_buf[k];
            }
    } // }}}

    ///////////////////////////////////////////////////////////////////////////

    Real initial_state(IntVect here) const
    { // {{{
        return this->analytic_solution(here, 0.0);
    } // }}}

    Real analytic_solution(IntVect here, Real t) const
    { // {{{
        Real x, y, z;
        x = center_coords(here)[0];
        y = center_coords(here)[1];
        z = center_coords(here)[2];

        Real const cx_pi = cx*M_PI;
        Real const cy_pi = cy*M_PI;
        Real const cz_pi = cz*M_PI;

        Real const omega_xy = -cx_pi*vx - cy_pi*vy; 
        Real const omega_z  = cz_pi*cz_pi*kz;
        return std::sin(cx_pi*x + cy_pi*y + omega_xy*t)
             + std::exp(-omega_z*t)*std::cos(cz_pi*z);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////

    std::string print_csv_header() const
    { // {{{
        return "X Velocity (vx),"
               "Y Velocity (vy),"
               "Z Diffusion Coefficient (kz)";
    } // }}}

    CSVTuple<Real, Real, Real> print_csv() const
    { // {{{
        return StreamCSV(vx, vy, kz);
    } // }}}

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    { // {{{
        ar & config;
        ar & cx;
        ar & cy;
        ar & cz;
        ar & vx;
        ar & vy;
        ar & kz;
    } // }}}

    configuration config;

  private:
    Real cx;
    Real cy;
    Real cz;

  public:
    Real vx;
    Real vy;
    Real kz;
};

}

#endif // CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2

