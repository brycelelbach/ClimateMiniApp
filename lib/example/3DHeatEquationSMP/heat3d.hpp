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
#include "FluxBox.H"

#include <cstdint>
#include <cmath>

#include <tuple>
#include <vector>

#include <lapacke.h>

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

bool upper_boundary(boundary_type bdry)
{
    switch (bdry)
    {
        case LOWER_X:
        case LOWER_Y:
        case LOWER_Z:
            return false;
        case UPPER_X:
        case UPPER_Y:
        case UPPER_Z:
            return true;
    };

    assert(false);
    return false;
}

bool lower_boundary(boundary_type type)
{
    return !upper_boundary(type);
}

struct problem_state
{
    problem_state()
      : reflux_horizontal_needed(false)
      , reflux_vertical_needed(false)
    {}

    // TODO: Replace with constructor.
    void define(problem_state const& solution)
    {
        U.define(solution.U.disjointBoxLayout()
               , solution.U.nComp()
               , solution.U.ghostVect());
        F.define(solution.U.disjointBoxLayout()
               , solution.U.nComp()
               , IntVect::Zero); 
    }

    // TODO: Replace with constructor.
    void alias(LevelData<FArrayBox>& A)
    {
        Interval alias_int(0, A.nComp()-1);
        aliasLevelData<FArrayBox>(U, &A, alias_int);
        F.define(U.disjointBoxLayout()
               , U.nComp()
               , U.ghostVect());
    }

    void copy(problem_state const& A)
    {
        define(A);
        DataIterator dit = A.U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            U[dit].copy(A.U[dit]);
            F[dit].copy(A.F[dit]);
        }
    }

    void zero()
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            U[dit].setVal(0.0);
            F[dit].setVal(0.0);
        }
    }

    void abs()
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            U[dit].abs();
            reflux_horizontal_needed = true;    
            reflux_vertical_needed = true;    
        }
    }

    Real sum() const
    {
        Real tmp = 0.0;

        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            tmp += U[dit].sum(0);

        return tmp;
    }

    void increment(problem_state const& A, Real factor = 1.0)
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            U[dit].plus(A.U[dit], factor);
            reflux_horizontal_needed = true;    
            reflux_vertical_needed = true;    
        }
    }

    LevelData<FArrayBox> const& data() const
    {
        return U;
    }

    LevelData<FArrayBox>& data()
    {
        return U;
    }

    LevelData<FluxBox> const& flux() const
    {
        return F;
    }

    LevelData<FluxBox>& flux()
    {
        return F;
    }

    IntVect ghosts() const
    {
        return U.ghostVect();
    }

    // TODO: Move call to imexop
    void exchange()
    {
        U.exchange();
    }

    bool reflux_horizontal_needed; 
    bool reflux_vertical_needed;

  private:
    /// Data for this level
    LevelData<FArrayBox> U;

    /// Fluxes
    LevelData<FluxBox> F;
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
        // FIXME: Temporary hackage.
        profile.reflux_horizontal(const_cast<problem_state&>(phi_));

        DataIterator dit = kE_.data().dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        { 
            auto& kE = kE_.data()[dit];
            auto const& phi = phi_.data()[dit];
            auto const& phi_FY = phi_.flux()[dit].getFlux(1);
            auto const& phi_FZ = phi_.flux()[dit].getFlux(2);
 
            IntVect lower = phi.smallEnd();
            IntVect upper = phi.bigEnd();
    
            ///////////////////////////////////////////////////////////////////
            // Horizontal BCs
            auto BCs =
                [&] (boundary_type type, size_t dir, IntVect V)
                { 
                    int sign = (upper_boundary(type) ? -1 : 1);
 
                    if (profile.is_outside_domain(type, V[dir]))
                    {
                        size_t A, B;
        
                        if      (0 == dir) { A = 1; B = 2; }
                        else if (1 == dir) { A = 0; B = 2; }
                        else if (2 == dir) { A = 0; B = 1; }
                        else    assert(false);
 
                        for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                            for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                            {
                                IntVect outside(V);
                                outside.setVal(A, a);
                                outside.setVal(B, b);
       
                                for (int c = 0; c <= phi_.ghosts()[dir]; ++c)
                                {
                                    outside.shift(dir, sign*c);
                                    kE(outside) = profile.outside_domain(type, outside, phi, t); 
                                } 
                            }

                        //int sign = (upper_boundary(type) ? -1 : 1);
                        //V.shift(dir, sign*phi_.ghosts()[dir]);
                    }

//                    if (profile.is_boundary(type, (V+sign*phi_.ghosts())[dir]))
                    if (profile.is_boundary(type, (V)[dir]))
                    {
                        size_t A, B;
        
                        if      (0 == dir) { A = 1; B = 2; }
                        else if (1 == dir) { A = 0; B = 2; }
                        else if (2 == dir) { A = 0; B = 1; }
                        else    assert(false);
 
                        for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                            for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                            {
                                IntVect bdry(V);
                                bdry.shift(dir, sign*phi_.ghosts()[dir]);
                                bdry.setVal(A, a);
                                bdry.setVal(B, b);

                                kE(bdry) = profile.boundary_conditions(type, bdry, phi, t); 
                            }

                        int sign = (upper_boundary(type) ? -1 : 1);
                        V.shift(dir, sign*IntVect::Unit[dir]);
                    }

                    return V;
                };

            lower = BCs(LOWER_X, 0, lower);
            upper = BCs(UPPER_X, 0, upper);
            lower = BCs(LOWER_Y, 1, lower);
            upper = BCs(UPPER_Y, 1, upper);
            lower = BCs(LOWER_Z, 2, lower);
            upper = BCs(UPPER_Z, 2, upper);

            lower.shift(phi_.ghosts());
            upper.shift(-1*phi_.ghosts());

/*
            if (profile.is_boundary(UPPER_X, upper[0]))
                for (auto j = lower[1]; j <= upper[1]; ++j)
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect upper_x(upper[0], j, k);
                        kE(upper_x) = profile.horizontal_bcs(UPPER_X, upper_x, phi, t); 
                    }

            if (profile.is_boundary(LOWER_Y, lower[1]))
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect lower_y(i, lower[1], k);
                        kE(lower_y) = profile.horizontal_bcs(LOWER_Y, lower_y, phi, t); 
                    }

            if (profile.is_boundary(UPPER_Y, upper[1]))
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect upper_y(i, upper[1], k);
                        kE(upper_y) = profile.horizontal_bcs(UPPER_Y, upper_y, phi, t); 
                    }
    
            if (profile.is_boundary(LOWER_Z, lower[2]))
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    for (auto j = lower[1]; j <= upper[1]; ++j)
                    {
                        IntVect lower_z(i, j, lower[2]);
                        kE(lower_z) = profile.horizontal_bcs(LOWER_Z, lower_z, phi, t); 
                    }

            if (profile.is_boundary(UPPER_Z, upper[2]))
                for (auto i = lower[0]; i <= upper[0]; ++i)
                    for (auto j = lower[1]; j <= upper[1]; ++j)
                    {
                        IntVect upper_z(i, j, upper[2]);
                        kE(upper_z) = profile.horizontal_bcs(UPPER_Z, upper_z, phi, t); 
                    }
*/
    
            ///////////////////////////////////////////////////////////////////
            // Interior points.
            for (auto i = lower[0]; i <= upper[0]; ++i)
                for (auto j = lower[1]; j <= upper[1]; ++j)
                    for (auto k = lower[2]; k <= upper[2]; ++k)
                    {
                        IntVect here(i, j, k);

//                        if (profile.source_term(here) < 0.0)
//                            std::cout << here << " has a negative source term\n";
                        //std::cout << "interior:" << here << "\n";

                        kE(here) = profile.horizontal_stencil(here, phi_FY, phi_FZ)
                                 + profile.source_term(here); 
                    }
        }
    }

    void implicitOp(problem_state& kI, problem_state const& phi, Real t)
    {
        kI.zero();
    }

    void solve(problem_state& phi_, problem_state const& rhs_, Real t, Real dtscale)
    {
        // FIXME FIXME FIXME: I hate this, LAPACK can operate in place, can
        // we avoid this copy?
//        phi_.copy(rhs_);

        DataIterator dit = rhs_.data().dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        { 
            auto& phi = phi_.data()[dit];
            auto const& rhs = rhs_.data()[dit];
 
            IntVect lower = rhs.smallEnd();
            IntVect upper = rhs.bigEnd();

            auto BCs =
                [&] (boundary_type type, size_t dir, IntVect V)
                { 
                    if (profile.is_boundary(type, V[dir]))
                    {
                        size_t A, B;
        
                        if      (0 == dir) { A = 1; B = 2; }
                        else if (1 == dir) { A = 0; B = 2; }
                        else if (2 == dir) { A = 0; B = 1; }
                        else    assert(false);
 
                        int sign = (upper_boundary(type) ? -1 : 1);
                        V.shift(dir, sign*IntVect::Unit[dir]);
                    }

                    return V;
                };

            lower = BCs(LOWER_X, 0, lower);
            upper = BCs(UPPER_X, 0, upper);
            lower = BCs(LOWER_Y, 1, lower);
            upper = BCs(UPPER_Y, 1, upper);
            lower = BCs(LOWER_Z, 2, lower);
            upper = BCs(UPPER_Z, 2, upper);

            lower.shift(phi_.ghosts());
            upper.shift(-1*phi_.ghosts());

            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto k = lower[2]; k <= upper[2]; ++k)
                {
                    auto A = profile.vertical_operator(j, k, phi, dtscale);

                    // TODO: give this a return value.
                    profile.vertical_solve(j, k, A, phi);
                }
        }
    }

  private:
    Profile profile;
};

// FIXME: here for debugging
}

void output(FArrayBox const& soln, std::string const& format, std::string const& name, std::uint64_t step);
void output(LevelData<FArrayBox> const& data, std::string const& format, std::string const& name, std::uint64_t step);
void output(heat3d::problem_state const& soln, std::string const& format, std::string const& name, std::uint64_t step);

namespace heat3d {

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
        Real constexpr CFL = 0.4;
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
        return std::sin(A*M_PI*x)*std::sin(B*M_PI*y)*std::sin(C*M_PI*z);
    } 

    Real horizontal_stencil(IntVect here, FArrayBox const& phi_FY, FArrayBox const& phi_FZ)
    {
        IntVect north(here[0], here[1], here[2]+1);
        IntVect south(here[0], here[1], here[2]);
        IntVect east (here[0], here[1]+1, here[2]);
        IntVect west (here[0], here[1], here[2]);

        Real const dh = std::get<1>(dp()); 
        return (-1.0/dh) * (phi_FY(east) - phi_FY(west) + phi_FZ(north) - phi_FZ(south)); 
    }

//    Real horizontal_kernel(Real dh2, Real here, Real north, Real south, Real east, Real west, bool debug = false)
    Real horizontal_kernel(Real dh2, Real here, Real north, Real south, Real east, Real west)
    {
/*
        if (debug)
        std::cout << "***\n"
                  << "dh2: " << dh2
                  << " here: " << here 
                  << " north: " << north
                  << " south: " << south
                  << " east: " << east 
                  << " west: " << west << "\nval:" << ((1.0/dh2) * (ky*(east+west) + kz*(north+south) - 4.0*here)) << "\n"
                ;
*/
//        return (1.0/dh2) * (ky*(east+west) + kz*(north+south) - 4.0*here);
        return (1.0/dh2) * (ky*(east+west) + kz*(north+south) - 2.0*(ky+kz)*here);
    }

    // TODO: This should have a corresponding "kernel" function so that dependencies
    // are properly expressed.
    Real boundary_conditions(boundary_type bdry, IntVect here, FArrayBox const& phi, Real t)
    {
        // U_t(t,0,y,z) = U_t(t,1,y,z) = U_t(t,x,0,z) = U_t(t,x,1,z) = 0
        return 0.0;//source_term(here);
    } 

    Real outside_domain(boundary_type bdry, IntVect here, FArrayBox const& phi, Real t)
    {
        // U_t(t,0,y,z) = U_t(t,1,y,z) = U_t(t,x,0,z) = U_t(t,x,1,z) = 0
        return 0.0;//source_term(here);
    } 

    typedef std::tuple<std::vector<Real>, std::vector<Real>, std::vector<Real> >
        crs_matrix;
    
    // TODO: Could be cached.
    crs_matrix vertical_operator(int j, int k, FArrayBox const& phi, Real dtscale)
    {
        std::uint64_t const size = phi.size()[0];
        Real const dv = std::get<0>(dp()); 
        Real const kvdv = kx/(dv*dv); 
        Real const H = dt()*dtscale;

//        std::cout << size << std::endl;

        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(size-1, H*(kvdv/2.0));
        // Diagonal part of the matrix.
        std::vector<Real> d(size, 1.0-H*kvdv);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(size-1, H*(kvdv/2.0));

        ///////////////////////////////////////////////////////////////////////
        // Vertical BCs
        // TODO: Move these out or move the horizontal BCs into the profile.
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();

/*
        if (is_boundary(LOWER_X, lower[0]))
        {
            du.back() = 0.0;
            d.back() = 1.0;
            dl.back() = 0.0;
        }

        if (is_boundary(UPPER_X, upper[0]))
        {
            du.front() = 0.0;
            d.front() = 1.0;
            dl.front() = 0.0;
        }
*/

        return crs_matrix(dl, d, du); 
    }

    // TODO: Give this a return value.
    void vertical_solve(int j, int k, crs_matrix& A, FArrayBox& phi)
    {
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();

        // This is why we've picked 'x' as our vertical dimension; x columns
        // are contiguous in memory.

        assert((&phi(IntVect(lower[0]+1, j, k)) - &phi(IntVect(lower[0], j, k))) == 1);

        //phi(IntVect(lower[0], j, k)) = 0.0;
        //phi(IntVect(upper[0], j, k)) = 0.0;

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
    }

    bool is_outside_domain(boundary_type bdry, int l)
    {
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
            case UPPER_X:
                return (l == -1) || (l == config.nv);

            // Horizontal
            case LOWER_Y:
            case UPPER_Y:
            case LOWER_Z:
            case UPPER_Z:
                return (l == -1) || (l == config.nh);
        };

        assert(false);
        return false;
    } 

    bool is_outside_domain(IntVect here)
    {
        return is_outside_domain(LOWER_X, here[0])
            || is_outside_domain(LOWER_Y, here[1])
            || is_outside_domain(LOWER_Z, here[2]);
    }

    bool is_boundary(boundary_type bdry, int l)
    {
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
            case UPPER_X:
                return (l == 0) || (l == config.nv-1);

            // Horizontal
            case LOWER_Y:
            case UPPER_Y:
            case LOWER_Z:
            case UPPER_Z:
                return (l == 0) || (l == config.nh-1);
        };

        assert(false);
        return false; 
    } 

    bool is_boundary(IntVect here)
    {
        return is_boundary(LOWER_X, here[0])
            || is_boundary(LOWER_Y, here[1])
            || is_boundary(LOWER_Z, here[2]);
    }

    Real initial_state(IntVect here)
    {
        return 0.0;
    }

    Real exact_solution(IntVect here, Real t) 
    {
        Real x, y, z; std::tie(x, y, z) = phys_coords(here);
        if (is_outside_domain(here)) return 0.0;
        else if (is_boundary(here)) return 0.0;
        return exact_solution(x, y, z, t);
    }

    Real exact_solution(Real x, Real y, Real z, Real t) 
    {
        Real const K = (A*A*kx + B*B*ky + C*C*kz)*M_PI*M_PI;
        Real const a = (1.0 - std::exp(-t*K)) / (K); 
        return a * source_term(x, y, z);
    } 

    void reflux_horizontal(problem_state& soln)
    {
        static std::uint64_t i = 0;
        output(soln, std::string("reflux.%06u.hdf5"), std::string("phi"), ++i);

        DataIterator dit = soln.data().dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        { 
            auto const& U = soln.data()[dit];
            auto& F = soln.flux()[dit];

            auto& FY = F.getFlux(1);
            auto& FZ = F.getFlux(2); 

            IntVect lower = U.smallEnd();
            IntVect upper = U.bigEnd(); 

            lower.shift(soln.ghosts());
            upper.shift(-1*soln.ghosts());

            FY.setVal(0.0);
            FZ.setVal(0.0);

//            std::cout << lower << " "
//                      << upper << std::endl;
            for (auto k = lower[2]; k <= upper[2]; ++k)
                for (auto j = U.smallEnd()[1]+1; j <= U.bigEnd()[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        // NOTE: This currently assumes constant-coefficients 
                        // and assumes we're just taking an average of the
                        // coefficients. For variable-coefficient we need
                        // to actually take an average. 
                        // 
                        // F_{j-1/2} ~= -k_{j-1/2}/h (U_j - U_{j-1})

                        Real const dh = std::get<1>(dp()); 

                        IntVect U_left_y(i, j-1, k  );

                        IntVect U_here(i,   j,   k  );
                        IntVect F_here(i, j, k);

//                        if (U(U_here) > 1e200 || U(U_left_y) > 1e200 || U(U_left_z) > 1e200)
//                            std::cout << U_here << " " << U(U_here) << " " << U(U_left_y) << " " << U(U_left_z) << "\n";

//                        if (U(U_here) > 1e200 || U(U_left_y) > 1e200)
//                            std::cout << "y: " << U_here << " " << U(U_here) << " " << U(U_left_y) << "\n";

                        FY(F_here) = -(ky/dh) * (U(U_here) - U(U_left_y));

//                        if (FY(F_here) > 1e200 || FZ(F_here) > 1e200)
//                            std::cout << F_here << " " << FY(F_here) << " " << FZ(F_here) << "\n";
                    }

            for (auto k = U.smallEnd()[2]+1; k <= U.bigEnd()[2]; ++k)
                for (auto j = lower[1]; j <= upper[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        // NOTE: This currently assumes constant-coefficients 
                        // and assumes we're just taking an average of the
                        // coefficients. For variable-coefficient we need
                        // to actually take an average. 
                        // 
                        // F_{j-1/2} ~= -k_{j-1/2}/h (U_j - U_{j-1})

                        Real const dh = std::get<1>(dp()); 

                        IntVect U_left_z(i, j,   k-1);

                        IntVect U_here(i,   j,   k  );
                        IntVect F_here(i, j, k);

//                        if (U(U_here) > 1e200 || U(U_left_z) > 1e200)
//                            std::cout << "z: " << U_here << " " << U(U_here) << " " << U(U_left_z) << "\n";

                        FZ(F_here) = -(kz/dh) * (U(U_here) - U(U_left_z));

//                        if (FY(F_here) > 1e200 || FZ(F_here) > 1e200)
//                            std::cout << F_here << " " << FY(F_here) << " " << FZ(F_here) << "\n";
                    }

//            static std::uint64_t P = 0;
//            output(FY, std::string("flux.Y.%06u.hdf5"), std::string("fluxY"), ++P);
//            output(FZ, std::string("flux.Z.%06u.hdf5"), std::string("fluxZ"), P);
        }

        soln.reflux_horizontal_needed = false;
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

