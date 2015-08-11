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

#include "CMAConfiguration.H"
#include "CMAProblemState.H"
#include "CMAImExOperators.H"
#include "CMAProfileBase.H"

/*
#if defined(__INTEL_COMPILER)
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif
*/

namespace climate_mini_app
{

struct advection_diffusion_profile : profile_base<advection_diffusion_profile>
{
    typedef profile_base<advection_diffusion_profile> base_type;

    advection_diffusion_profile(
        configuration config_
      , Real cx_, Real cy_, Real cz_
      , Real kx_, Real vy_, Real vz_
        )
    // {{{
      : base_type(config_)
      , cx(cx_), cy(cy_), cz(cz_)
      , kx(kx_), vy(vy_), vz(vz_)
    {} // }}}

    // Time step size
    Real dt() const
    { // {{{
        // Check if advection is on. 
        // (vy!=0) || (vz!=0)
        if ((std::fabs(vy-0.0) > 1e-14) || (std::fabs(vz-0.0) > 1e-14))
        {
            // For advection: 
            //
            //      dt = CFL*dh/(vy+vz) 
            // 
            //      with CFL < 1.0 
            // 
            Real constexpr CFL = 0.8;
            Real const dh = dp()[1]; 

            return CFL*dh/(std::abs(vy)+std::abs(vz));
        }

        // Diffusion only.
        else
        {
            // We want something with the same magnitude as dv.
            Real constexpr A = 0.8;
            Real const dv = dp()[0];

            return A*dv;
        }
    } // }}}

    ProblemDomain problem_domain() const
    { // {{{
        IntVect lower_bound(IntVect::Zero);
        IntVect upper_bound(
            (config.nv-1),
            (config.nh-1),
            (config.nh-1)
        );

        bool is_periodic[] = { true, true, true };
        ProblemDomain domain(lower_bound, upper_bound, is_periodic);

        return domain;
    } // }}} 
    
    template <std::size_t Dir>
    Real horizontal_flux_stencil(
        IntVect here
      , FArrayBox const& phi
        ) const
    { // {{{
        Real const dh = dp()[1]; 

        Real v = 0.0;
        IntVect left;

        if      (1 == Dir)
        { 
            v = vy;
            left = IntVect(here[0], here[1]-1, here[2]);
        }

        else if (2 == Dir)
        {
            v = vz;
            left = IntVect(here[0], here[1], here[2]-1);
        }

        else
            assert(false);

        if (!phi.box().contains(left))
            return 0.0;

        return v * 0.5 * (phi(left) + phi(here));
    } // }}}

    Real inverse_dh_c() const
    {
        Real const dh = dp()[1]; 

        #if defined(CH_LOWER_ORDER_EXPLICIT_STENCIL)
            return (-1.0/dh);
        #else
            Real constexpr c = (-1.0/60.0);
            return c*(1.0/dh); 
        #endif
    }

    // FIXME: IntVect indexing may be expensive, compare to Hans' loops
    Real horizontal_stencil(
        IntVect here
      , FArrayBox const& FY
      , FArrayBox const& FZ
      , Real inv_dh_c
        ) const
    { // {{{ 
        return horizontal_stencil_1d<1>(here, FY, inv_dh_c)
             + horizontal_stencil_1d<2>(here, FZ, inv_dh_c);
    } // }}}

    // FIXME - should be a tiled stencil loop
    template <std::size_t Dir>
    Real horizontal_stencil_1d(
        IntVect here
      , FArrayBox const& F
      , Real inv_dh_c
        ) const
    { // {{{
        static_assert((Dir == 1) || (Dir == 2), "Illegal Dir, should be 1 or 2");

        #if defined(CH_LOWER_ORDER_EXPLICIT_STENCIL)
            IntVect right1(here); right1.setVal(Dir, here[Dir]+1);
    
            return inv_dh_c * (F(right1) - F(here)); 
        #else
            IntVect left2(here); left2.setVal(Dir, here[Dir]-2);
            IntVect left1(here); left1.setVal(Dir, here[Dir]-1);
    
            IntVect right1(here); right1.setVal(Dir, here[Dir]+1);
            IntVect right2(here); right2.setVal(Dir, here[Dir]+2);
            IntVect right3(here); right3.setVal(Dir, here[Dir]+3);

            return inv_dh_c
                 * ( 3.0*F(left2) - 30.0*F(left1)
                   - 20.0*F(here)
                   + 60.0*F(right1) - 15.0*F(right2) + 2.0*F(right3)
                   ); 
        #endif
    } // }}}

    void build_vertical_operator_for_solve(
        LapackFactorization& A
      , Box b
      , Real dtscale
        ) const
    { // {{{
        if (std::fabs(kx-0.0) < 1e-14)
            return;

        Real const c1 = 1.0;
        Real const c2 = -(kx*dt()*dtscale);
        build_vertical_operator(A, b, c1, c2);
    } // }}}

    void build_vertical_operator_for_apply(
        LapackFactorization& A
      , Box b
        ) const
    { // {{{
        if (std::fabs(kx-0.0) < 1e-14)
            return;

        Real const c1 = 0.0;
        Real const c2 = kx;
        build_vertical_operator(A, b, c1, c2);
    } // }}}

    void build_vertical_operator(
        LapackFactorization& A
      , Box b
      , Real c1
      , Real c2 
        ) const
    { // {{{
        std::uint64_t const size = b.size()[0];
        Real const dv = dp()[0]; 

        Real const coef = c2/(12.0*dv*dv); // for 4th-order
        Real const s[]={0.0, -1.0, 16.0, -30.0, 16.0, -1.0, 0.0};

        int const KU = 3;
        int const KL = 3;

        A.define(size, KL, KU);
        A.setZero();
        
        int ncol = A.numCols();
        for (int col=0; col < ncol; col++) 
            for (int ix=-KU; ix <= KL; ix++)
            {
                int row = col + ix;
                if ((row >= 0) && (row < ncol))
                    A(row, col) = ((row == col) ? c1 : 0) + s[ix+KU]*coef;
            }
        
        // Fix the diagonal ends for the homogeneous Neumann bc's
        // For 4th-order operator

        // The gradient flux for the first interior face
        int const lenG = 4;
        Real const sG1[] = {-145.0, 159.0, -15.0, 1.0};
        Real const coefG1 = c2/(120*dv*dv);

        // The gradient flux for the second interior (regular) face
        Real const sG2[] = {1.0, -15.0, 15.0, -1.0};
        Real const coefG2 = c2/(12*dv*dv);

        for (int ix=0; ix < lenG; ix++)
        {
            A(0,ix) = ((ix == 0) ? c1 : 0) + sG1[ix] * coefG1;
            A(1,ix) = ((ix == 1) ? c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;

            int ixflip = ncol-ix-1;
            A(ncol-1,ixflip) = ((ix == 0) ? c1 : 0) + sG1[ix] * coefG1;
            A(ncol-2,ixflip) =
                ((ix == 1) ? c1 : 0) + sG2[ix] * coefG2 - sG1[ix] * coefG1;
        }
    } // }}}

    void apply_vertical_operator(
        int j
      , int k
      , LapackFactorization& A
      , Box b
      , FArrayBox& kI
      , FArrayBox& phi
        ) const
    { // {{{
        if (std::fabs(kx-0.0) < 1e-14)
            return;

        IntVect lower = b.smallEnd();
        IntVect upper = b.bigEnd();

        // We've picked 'x' as our vertical dimension so x columns are
        // contiguous in memory; this check tests that.
        auto p0 = &phi(IntVect(lower[0], j, k));
        auto p1 = &phi(IntVect(lower[0]+1, j, k));
        assert((p1 - p0) == 1);

        Real* state = &phi(IntVect(lower[0], j, k)); 
        Real* rhs   = &kI(IntVect(lower[0], j, k)); 

        LapackWrapper::applyBandMatrix(state, rhs, A);
    } // }}}

    void vertical_solve(
        int j
      , int k
      , LapackFactorization& A
      , Box b
      , FArrayBox& phi
        ) const
    { // {{{
        if (std::fabs(kx-0.0) < 1e-14)
            return;

        IntVect lower = b.smallEnd();
        IntVect upper = b.bigEnd();

        // We've picked 'x' as our vertical dimension so x columns are
        // contiguous in memory; this check tests that.
        auto p0 = &phi(IntVect(lower[0], j, k));
        auto p1 = &phi(IntVect(lower[0]+1, j, k));
        assert((p1 - p0) == 1);

        Real* rhs = &phi(IntVect(lower[0], j, k)); 

        LapackFactorization AB;
        AB.define(A);

        LapackWrapper::factorBandMatrix(AB);
        LapackWrapper::solveBandMatrix(AB, rhs);
    } // }}}

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

        Real const omega_x = cx_pi*cx_pi*kx;
        Real const omega_yz = -cz_pi*vz - cy_pi*vy; 
        return std::exp(-omega_x*t)*std::cos(cx_pi*x)
             + std::sin(cy_pi*y + cz_pi*z + omega_yz*t);
    } // }}}

    Real source_term(IntVect here, Real t) const
    { // {{{
        return 0.0; 
    } // }}}

    std::string print_csv_header() const
    { // {{{
        return "X Diffusion Coefficient (kx),"
               "Y Velocity (vy),"
               "Z Velocity (vz)";
    } // }}}

    CSVTuple<Real, Real, Real> print_csv() const
    { // {{{
        return StreamCSV(kx, vy, vz);
    } // }}}

  private:
    Real const cx;
    Real const cy;
    Real const cz;

  public:
    Real const kx;
    Real const vy;
    Real const vz;
};

}

#endif // CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2

