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

#if defined(__INTEL_COMPILER)
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

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
        if ((std::fabs(vy-0.0) > 1e-16) || (std::fabs(vz-0.0) > 1e-16))
        {
            // For advection: 
            //
            //      dt = CFL*dh/(vy+vz) 
            // 
            //      with CFL < 1.0 
            // 
            Real constexpr CFL = 0.8;
            Real const dh = std::get<1>(dp()); 

            return CFL*dh/(std::abs(vy)+std::abs(vz));
        }

        // Diffusion only.
        else
        {
            // We want something with the same magnitude as dv.
            Real constexpr A = 0.8;
            Real const dv = std::get<0>(dp());

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
    
    Real horizontal_flux_stencil(
        IntVect here
      , std::size_t dir
      , FArrayBox const& phi
        ) const
    { // {{{
        Real const dh = std::get<1>(dp()); 

        Real v = 0.0;
        IntVect left;

        if      (1 == dir)
        { 
            v = vy;
            left = IntVect(here[0], here[1]-1, here[2]);
        }

        else if (2 == dir)
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

    // FIXME: IntVect indexing may be expensive, compare to Hans' loops
    Real horizontal_stencil(
        IntVect here
      , FArrayBox const& FY
      , FArrayBox const& FZ
        ) const
    { // {{{ 
        return horizontal_stencil_1d(here, 1, FY)
             + horizontal_stencil_1d(here, 2, FZ);
    } // }}}

    Real horizontal_stencil_1d(
        IntVect here
      , std::size_t dir
      , FArrayBox const& F
        ) const
    { // {{{
        assert((dir == 1) || (dir == 2));

        #if defined(CH_LOWER_ORDER_EXPLICIT_STENCIL)
            IntVect right1(here); right1.setVal(dir, here[dir]+1);
    
            Real const dh = std::get<1>(dp()); 
            return (-1.0/dh) * (F(right1) - F(here)); 
        #else
            IntVect left3(here); left3.setVal(dir, here[dir]-3);
            IntVect left2(here); left2.setVal(dir, here[dir]-2);
            IntVect left1(here); left1.setVal(dir, here[dir]-1);
    
            IntVect right1(here); right1.setVal(dir, here[dir]+1);
            IntVect right2(here); right2.setVal(dir, here[dir]+2);
            IntVect right3(here); right3.setVal(dir, here[dir]+3);
    
            Real constexpr alpha = -12.0;
    
            Real constexpr a        = 0.0 - (5.0*alpha)/3.0;   // a_{i}
    
            Real constexpr a_left1  = -45.0 + (5.0*alpha)/4.0; // a_{i-1}
            Real constexpr a_right1 = +45.0 + (5.0*alpha)/4.0; // a_{i+1}
    
            Real constexpr a_left2  = +9.0 - alpha/2.0;        // a_{i-2}
            Real constexpr a_right2 = -9.0 - alpha/2.0;        // a_{i+2}
    
            Real constexpr a_left3  = -1.0 + alpha/12.0;       // a_{i-3}
            Real constexpr a_right3 = +1.0 + alpha/12.0;       // a_{i+3}
    
            Real const dh = std::get<1>(dp()); 
            return (-1.0/(60.0*dh))
                 * ( -a_right3*F(left3) + -a_right2*F(left2) + -a_right1*F(left1)
                   + -a*F(here)
                   + -a_left1*F(right1) + -a_left2*F(right2) + -a_left3*F(right3)
                   ); 
        #endif
    } // }}}

    typedef std::tuple<std::vector<Real>, std::vector<Real>, std::vector<Real> >
        crs_matrix;
    
    crs_matrix vertical_operator(
        int j
      , int k
      , FArrayBox const& phi
      , Real dtscale
      , Box b
        ) const
    { // {{{
        std::uint64_t const size = b.size()[0];
        Real const dv = std::get<0>(dp()); 
        Real const beta = (kx*(dt()*dtscale))/(dv*dv);

        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(size-1, -beta);
        // Diagonal part of the matrix.
        std::vector<Real> d(size, 1.0+2.0*beta);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(size-1, -beta);

        // No-flux boundaries.
        du.front() = -2.0*beta;
        dl.back()  = -2.0*beta;

        return crs_matrix(dl, d, du); 
    } // }}}

    void vertical_solve(
        int j
      , int k
      , crs_matrix& A
      , FArrayBox& phi
      , Box b
        ) const
    { // {{{
        if (std::fabs(kx-0.0) < 1e-16)
            return;

        IntVect lower = b.smallEnd();
        IntVect upper = b.bigEnd();

        // We've picked 'x' as our vertical dimension so x columns are
        // contiguous in memory; this check tests that.
        auto p0 = &phi(IntVect(lower[0], j, k));
        auto p1 = &phi(IntVect(lower[0]+1, j, k));
        assert((p1 - p0) == 1);

        Real* rhs = &phi(IntVect(lower[0], j, k)); 

        int info = LAPACKE_dgtsv(
            LAPACK_ROW_MAJOR, // matrix format
            std::get<1>(A).size(), // matrix order
            1, // # of right hand sides 
            std::get<0>(A).data(), // subdiagonal part
            std::get<1>(A).data(), // diagonal part
            std::get<2>(A).data(), // superdiagonal part
            rhs, // column to solve 
            1 // leading dimension of RHS
            );
 
        assert(info == 0);
    } // }}}

    Real initial_state(IntVect here) const
    { // {{{
        return this->analytic_solution(here, 0.0);
    } // }}}

    Real analytic_solution(IntVect here, Real t) const
    { // {{{
        Real x, y, z;
        x = std::get<0>(center_coords(here));
        y = std::get<1>(center_coords(here));
        z = std::get<2>(center_coords(here));

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

