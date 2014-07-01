/*
 *      _______              __ 
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2)
#define CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2

#include "REAL.H"
#include "LevelData.H"
#include "FArrayBox.H"

#include <cstdint>
#include <cmath>
#include <tuple>
#include <assert.h>

// Solving problems of the form:
//
//      U_t = kx * U_xx + ky * U_yy + kz * U_zz + h

namespace heat3d
{

struct configuration
{
    ///////////////////////////////////////////////////////////////////////////
    // Parameters.

    Real const nt; ///< Physical time to step to.

    std::uint64_t const nh; ///< "Horizontal" extent (y and z dimensions) per core.
    std::uint64_t const nv; ///< "Vertical" extent (x dimension) per core.

    std::uint64_t const max_box_size;

    ///////////////////////////////////////////////////////////////////////////

    configuration(
        Real nt_ 
      , std::uint64_t nh_
      , std::uint64_t nv_
      , std::uint64_t max_box_size_
        )
      : nt(nt_)
      , nh(nh_)
      , nv(nv_)
      , max_box_size(max_box_size_)
    {}
}; 

enum boundary_type
{
    // Vertical
    LOWER_X,
    UPPER_X,

    // Horizontal
    LOWER_Y,
    UPPER_Y,
    LOWER_Z,
    UPPER_Z
};

struct problem_state
{
    problem_state() {}

    // TODO: Replace with constructor.
    void define(problem_state const& solution)
    {
        U.define(solution.U.disjointBoxLayout()
               , solution.U.nComp()
               , solution.U.ghostVect());
    }

    // TODO: Replace with constructor.
    void alias(LevelData<FArrayBox>& A)
    {
        Interval alias_int(0, A.nComp()-1);
        aliasLevelData<FArrayBox>(U, &A, alias_int);
    }

    void copy(problem_state const& A)
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            U[dit].copy(A.U[dit]);
    }

    void zero()
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            U[dit].setVal(0.0);
    }

    void increment(problem_state const& A, Real factor = 1.0)
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            U[dit].plus(A.U[dit], factor);
    }

    LevelData<FArrayBox> const& data() const
    {
        return U;
    }

    LevelData<FArrayBox>& data()
    {
        return U;
    }

    void exchange()
    {
        U.exchange();
    }

  private:
    /// Data for this level
    LevelData<FArrayBox> U;
};

template <typename Profile>
struct imex_operators
{
    imex_operators(Profile const& profile_)
      : profile(profile_)
    {}

    void resetDt(Real)
    {
    }

    // TODO: Lift stencil.
    void explicitOp(problem_state& kE_, problem_state const& phi_, Real t)
    {
        DataIterator dit = kE_.data().dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        { 
            auto& kE = kE_.data()[dit];
            auto const& phi = phi_.data()[dit];
 
            IntVect lower = phi.smallEnd();
            IntVect upper = phi.bigEnd();
            Interval comps = phi.interval();
    
            ///////////////////////////////////////////////////////////////////
            // Horizontal BCs
            for (auto i = lower[0]; i <= upper[0]; ++i)
            {
                if (profile.is_horizontal_boundary(lower[1]))
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect lower_y(i, lower[1]  , k);
                        kE(lower_y) = profile.horizontal_bcs(LOWER_Y, lower_y, phi, t); 
                    }

                if (profile.is_horizontal_boundary(upper[1]))
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect upper_y(i, upper[1], k);
                        kE(upper_y) = profile.horizontal_bcs(UPPER_Y, upper_y, phi, t); 
                    }
    
                if (profile.is_horizontal_boundary(lower[2]))
                    for (auto j = lower[1]; j <= upper[1]; ++j)
                    {
                        IntVect lower_z(i, j, lower[2]  );
                        std::cout << lower_z << " " << phi(lower_z) << "\n";
                        kE(lower_z) = profile.horizontal_bcs(LOWER_Z, lower_z, phi, t); 
                    }

                if (profile.is_horizontal_boundary(upper[2]))
                    for (auto j = lower[1]; j <= upper[1]; ++j)
                    {
                        IntVect upper_z(i, j, upper[2]);
                        std::cout << upper_z << " " << phi(upper_z) << "\n";
                        kE(upper_z) = profile.horizontal_bcs(UPPER_Z, upper_z, phi, t); 
                    }
            }
    
            ///////////////////////////////////////////////////////////////////
            // Interior points.
            for (auto i = lower[0]; i <= upper[0]; ++i)
                for (auto j = lower[1]+1; j <= upper[1]-1; ++j)
                    for (auto k = lower[2]+1; k <= upper[2]-1; ++k)
                    {
                        IntVect here(i, j, k);

                        std::cout << "interior:" << here << "\n";

                        kE(here) = 1.0; 
                        //kE(here) = profile.horizontal_stencil(here, phi)
                        //         + profile.source_term(here); 
                    }
        }
    }

    void implicitOp(problem_state& kI, problem_state const& phi, Real t)
    {
        // no-op
        kI.zero();
    }

    void solve(problem_state& phi, problem_state const& rhs, Real t, Real dtscale)
    {
        phi.increment(rhs);
    }

  private:
    Profile profile;
};

// A simple, anisotropic profile:
//
//      U_t = kx * U_xx + ky * U_yy + kz * U_zz + h(x,y,z)     x,y,z in [0,1]
//
//      U(0,x,y,z) = 0
//      U_t(t,0,y,z) = U_t(t,1,y,z) = 0
//      U_t(t,x,0,z) = U_t(t,x,1,z) = 0
//      U_t(t,x,y,0) = U_t(t,x,y,1) = 0
//
//      Source: h(x,y,z) = sin(A*pi*x) * sin(B*pi*y) * sin(C*pi*z)
//
// This problem has an exact solution:
//
//      U(t,x,y) = a(t) * h(x,y,z)
//      a(t) = (1 - exp(-K*t)) / K
//      K = (A^2*kx + B^2*ky + C^2*kz)*pi^2
//
struct aniso_profile
{
    aniso_profile(
        configuration config_
      , Real A_, Real B_, Real C_
      , Real kx_, Real ky_, Real kz_
        )
      : config(config_)
      , A(A_), B(B_), C(C_)
      , kx(kx_), ky(ky_), kz(kz_)
    {}

    std::tuple<Real, Real, Real> phys_coords(IntVect here)
    {
        return std::tuple<Real, Real, Real>(
            Real(here[0])/(Real(config.nv)-1.0)
          , Real(here[1])/(Real(config.nh)-1.0)
          , Real(here[2])/(Real(config.nh)-1.0)
        );
    }

    // Spatial step size
    std::tuple<Real, Real, Real> dp()
    {
        return phys_coords(IntVect(1, 1, 1));
    }

    // Time step size
    Real dt()
    {
        // We only need to consider the horizontal dimensions, so we have: 
        //
        //      dt = min((CFL*(dy*dy))/ky, (CFL*(dz*dz))/kz)
        //
        // Conveniently, I've required dy == dz and ky/kz are constant for this
        // particular problem, so:
        Real constexpr CFL = 0.8;
        Real const dh = std::get<1>(dp()); 
        assert(ky > 0.0);
        assert(kz > 0.0);
        return (CFL*(dh*dh))/std::max(ky, kz);
    }

    Real source_term(IntVect here) 
    {
        Real x, y, z; std::tie(x, y, z) = phys_coords(here);
        return source_term(x, y, z);
    }

    Real source_term(Real x, Real y, Real z)
    {
        return /*std::sin(A*M_PI*x)*/std::sin(B*M_PI*y)*std::sin(C*M_PI*z);
    } 

    Real horizontal_stencil(IntVect here, FArrayBox const& phi)
    {
        IntVect north(here[0], here[1], here[2]+1);
        IntVect south(here[0], here[1], here[2]-1);
        IntVect east (here[0], here[1]+1, here[2]);
        IntVect west (here[0], here[1]-1, here[2]);

        Real const dh = std::get<1>(dp()); 

        return horizontal_kernel(dh*dh,
            phi(here), phi(north), phi(south), phi(east), phi(west));
    }

    Real horizontal_kernel(Real dh2, Real here, Real north, Real south, Real east, Real west)
    {
        return (1.0/dh2) * (ky*(east+west) + kz*(north+south) - 4.0*here);
    }


    // TODO: This should have a corresponding "kernel" function so that dependencies
    // are properly expressed.
    Real horizontal_bcs(boundary_type bdry, IntVect here, FArrayBox const& phi, Real t)
    {
        // U_t(t,0,y,z) = U_t(t,1,y,z) = U_t(t,x,0,z) = U_t(t,x,1,z) = 0
        return 0.0;
    } 

    bool is_horizontal_boundary(int yz)
    {
        return (yz == -1) || (yz == config.nh);
    } 

    Real initial_state(IntVect here)
    {
        return 0.0;
    }

    Real exact_solution(IntVect here, Real t) 
    {
        Real x, y, z; std::tie(x, y, z) = phys_coords(here);
        return exact_solution(t, x, y, z);
    }

    Real exact_solution(Real x, Real y, Real z, Real t) 
    {
        Real const K = (A*A*kx + B*B*ky + C*C*kz)*M_PI*M_PI;
        Real const a = (1.0 - std::exp(-t*K)) / K; 
        return a * source_term(x, y, z);
    } 

  private:
    configuration config;

    Real A;
    Real B;
    Real C;

  public:
    Real kx;
    Real ky;
    Real kz;
};

}

#endif // CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2

