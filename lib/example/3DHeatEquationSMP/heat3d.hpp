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

#include <hpx/lcos/when_all.hpp>

#include "REAL.H"
#include "LevelData.H"
#include "FArrayBox.H"
#include "FluxBox.H"

#include <cstdint>
#include <cmath>

#include <tuple>
#include <vector>

#include <assert.h>

#if defined(HPX_INTEL_VERSION)
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

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

    IntVect const ghost_vector;

    ///////////////////////////////////////////////////////////////////////////

    configuration(
        Real nt_ 
      , std::uint64_t nh_
      , std::uint64_t nv_
      , std::uint64_t max_box_size_
      , IntVect ghost_vector_
        )
      : nt(nt_)
      , nh(nh_)
      , nv(nv_)
      , max_box_size(max_box_size_)
      , ghost_vector(ghost_vector_)
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

struct streamBox
{
    Box const& b;
    int proc;

    streamBox(Box const& b_, int proc_ = -1) : b(b_), proc(proc_) {}

    friend std::ostream& operator<<(std::ostream& os, streamBox const& sb)
    {
        if (-1 == sb.proc) 
            return os << "(" << sb.b.smallEnd() << " " << sb.b.bigEnd()
                      << " " << sb.b.volume() << ")";
        else
            return os << "(L" << sb.proc
                      << " " << sb.b.smallEnd() << " " << sb.b.bigEnd()
                      << " " << sb.b.volume() << ")";
    } 
};

struct problem_state
{
    problem_state()
      : U() 
      , F() 
    {}

    void define(Box const& b, int comps)
    {
        data().define(b, comps); 
        flux().define(b, comps); 
    }

    void alias(problem_state& A)
    {
        data().define(A.data().interval(), A.data());
        flux().define(data().box(), A.nComp()); 
    }

    void copy(problem_state& src,
              Box const&     srcbox,
              int            srccomp,
              Box const&     destbox,
              int            destcomp,
              int            numcomp)
    {
/*
        std::cout << "SEND: "
                  << streamBox(src.data().box())
                  << " " << streamBox(srcbox) 
                  << " -> "
                  << streamBox(data().box())
                  << " " << streamBox(destbox) 
                  << "\n" << std::flush;
*/

        data().copy(src.data(), srcbox, srccomp, destbox, destcomp, numcomp);
    }

    void copy(problem_state const& A)
    {
        data().copy(A.data());
        flux().copy(A.flux());
    }

    int nComp() const
    {
        return data().nComp();
    }

    void zero()
    {
        data().setVal(0.0);
        flux().setVal(0.0);
    }

    void abs()
    {
        data().abs();
    }

    Real sum(int comp = 0) const
    {
        return data().sum(comp);
    }

    void increment(problem_state const& A, Real factor = 1.0)
    {
        data().plus(A.data(), factor);
    }

    FArrayBox const& data() const
    {
        return U;
    }

    FArrayBox& data()
    {
        return U; 
    }

    FluxBox& flux() const
    {
        return F;
    }

  private:
    FArrayBox U;

    /// Fluxes
    mutable FluxBox F;
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
        profile.reflux_horizontal(phi_);

        auto&       kE     = kE_.data();
        auto const& phi    = phi_.data();
        auto const& phi_FY = phi_.flux().getFlux(1);
        auto const& phi_FZ = phi_.flux().getFlux(2);
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();
    
        ///////////////////////////////////////////////////////////////////////
        // Horizontal BCs
        auto BCs =
            [&] (boundary_type type, size_t dir, IntVect V)
            { 
                int sign = (upper_boundary(type) ? -1 : 1);
 
                if (profile.is_outside_domain(type, V[dir]))
                {
                    size_t A = -1, B = -1;
        
                    if      (0 == dir) { A = 1; B = 2; }
                    else if (1 == dir) { A = 0; B = 2; }
                    else if (2 == dir) { A = 0; B = 1; }
                    else    assert(false);
 
                    for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                        for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                        {
                            IntVect out(V);
                            out.setVal(A, a);
                            out.setVal(B, b);
       
                            for (int c = 0; c <= profile.ghostVect()[dir]; ++c)
                            {
                                out.shift(dir, sign*c);
                                kE(out) = 
                                    profile.outside_domain(type, out, phi, t); 
                            } 
                        }
                }

                if (profile.is_boundary(type, (V+sign*profile.ghostVect())[dir]))
                {
                    size_t A = -1, B = -1;
        
                    if      (0 == dir) { A = 1; B = 2; }
                    else if (1 == dir) { A = 0; B = 2; }
                    else if (2 == dir) { A = 0; B = 1; }
                    else    assert(false);

                    for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                        for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                        {
                            IntVect bdry(V);
                            bdry.shift(dir, sign*profile.ghostVect()[dir]);
                            bdry.setVal(A, a);
                            bdry.setVal(B, b);

                            kE(bdry) =
                                profile.boundary_conditions(type, bdry, phi, t); 
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

        lower.shift(profile.ghostVect());
        upper.shift(-1*profile.ghostVect());
    
        ///////////////////////////////////////////////////////////////////////
        // Interior points.
        for (auto i = lower[0]; i <= upper[0]; ++i)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto k = lower[2]; k <= upper[2]; ++k)
                {
                    IntVect here(i, j, k);

                    kE(here) = profile.horizontal_stencil(here, phi_FY, phi_FZ)
                             + profile.source_term(here); 
                }
    }

    void implicitOp(problem_state& kI, problem_state const& phi, Real t)
    {
        kI.zero();
    }

    void solve(problem_state& phi_, Real t, Real dtscale)
    {
        auto& phi = phi_.data();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();

        ///////////////////////////////////////////////////////////////////////
        // Horizontal BCs
        auto BCs =
            [&] (boundary_type type, size_t dir, IntVect V)
            { 
                int sign = (upper_boundary(type) ? -1 : 1);
 
                if (profile.is_outside_domain(type, V[dir]))
                {
                    size_t A = -1, B = -1;
        
                    if      (0 == dir) { A = 1; B = 2; }
                    else if (1 == dir) { A = 0; B = 2; }
                    else if (2 == dir) { A = 0; B = 1; }
                    else    assert(false);
 
                    for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                        for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                        {
                            IntVect out(V);
                            out.setVal(A, a);
                            out.setVal(B, b);
       
                            for (int c = 0; c <= profile.ghostVect()[dir]; ++c)
                            {
                                out.shift(dir, sign*c);
                                phi(out) = 
                                    profile.outside_domain(type, out, phi, t); 
                            } 
                        }
                }

                if (profile.is_boundary(type, (V+sign*profile.ghostVect())[dir]))
                {
                    size_t A = -1, B = -1;
        
                    if      (0 == dir) { A = 1; B = 2; }
                    else if (1 == dir) { A = 0; B = 2; }
                    else if (2 == dir) { A = 0; B = 1; }
                    else    assert(false);

                    for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                        for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                        {
                            IntVect bdry(V);
                            bdry.shift(dir, sign*profile.ghostVect()[dir]);
                            bdry.setVal(A, a);
                            bdry.setVal(B, b);

                            phi(bdry) =
                                profile.boundary_conditions(type, bdry, phi, t); 
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

        lower.shift(profile.ghostVect());
        upper.shift(-1*profile.ghostVect());

        Box b(lower, upper);

/*
        std::vector<hpx::future<void> > vsolves;

        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k)
            {
                auto VSolve = [&](int j, int k)
                    {
                        auto A = profile.vertical_operator(j, k, phi, dtscale, b);

                        profile.vertical_solve(j, k, A, phi, b);
                    };

                vsolves.push_back(hpx::async(VSolve, j, k));
            }

        hpx::lcos::when_all(vsolves).get();
*/

        // FIXME: Was this made serial for performance reasons?
        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k)
            {
                auto A = profile.vertical_operator(j, k, phi, dtscale, b);

                profile.vertical_solve(j, k, A, phi, b);
            }
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
        // FIXME: Switch to cell centered
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
        Real x, y, z;
        x = std::get<0>(phys_coords(here));
        y = std::get<1>(phys_coords(here));
        z = std::get<2>(phys_coords(here));
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
        // IntVect indexing may be expensive, compare to Hans' loops
        return (-1.0/dh) * (phi_FY(east) - phi_FY(west) + phi_FZ(north) - phi_FZ(south)); 
    }

    Real boundary_conditions(boundary_type bdry, IntVect here, FArrayBox const& phi, Real t)
    {
        return 0.0;
    } 

    Real outside_domain(boundary_type bdry, IntVect here, FArrayBox const& phi, Real t)
    {
        return 0.0;
    } 

    typedef std::tuple<std::vector<Real>, std::vector<Real>, std::vector<Real> >
        crs_matrix;
    
    // TODO: Could be cached.
    crs_matrix vertical_operator(int j, int k, FArrayBox const& phi, Real dtscale, Box b)
    {
        std::uint64_t const size = b.size()[0];
        Real const dv = std::get<0>(dp()); 
        Real const kvdv = kx/(dv*dv); 
        Real const H = dt()*dtscale;

        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(size-1, H*(kvdv/2.0));
        // Diagonal part of the matrix.
        std::vector<Real> d(size, 1.0-H*kvdv);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(size-1, H*(kvdv/2.0));

        return crs_matrix(dl, d, du); 
    }

    // TODO: Give this a return value.
    void vertical_solve(int j, int k, crs_matrix& A, FArrayBox& phi, Box b)
    {
        IntVect lower = b.smallEnd();
        IntVect upper = b.bigEnd();

        // This is why we've picked 'x' as our vertical dimension; x columns
        // are contiguous in memory.

        assert((&phi(IntVect(lower[0]+1, j, k)) - &phi(IntVect(lower[0], j, k))) == 1);

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
        Real x, y, z;
        x = std::get<0>(phys_coords(here));
        y = std::get<1>(phys_coords(here));
        z = std::get<2>(phys_coords(here));
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

    void reflux_horizontal(problem_state const& soln)
    {
        auto const& U = soln.data();
        auto& F = soln.flux();

        auto& FY = F.getFlux(1);
        auto& FZ = F.getFlux(2); 

        IntVect lower = U.smallEnd();
        IntVect upper = U.bigEnd(); 

        lower.shift(ghostVect());
        upper.shift(-1*ghostVect());

// FIXME FIXME
//        FY.setVal(0.0);
//        FZ.setVal(0.0);

        auto FluxY = [&](IntVect lower, IntVect upper)
        {
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
    
    //                    if (U(U_here) > 1e200 || U(U_left_y) > 1e200)
    //                        std::cout << "y: " << U_here << " " << U(U_here) << " " << U(U_left_y) << "\n";
    
                        FY(F_here) = -(ky/dh) * (U(U_here) - U(U_left_y));
                    }
        };

        auto FluxZ = [&](IntVect lower, IntVect upper)
        {
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
    
    //                    if (U(U_here) > 1e200 || U(U_left_z) > 1e200)
    //                        std::cout << "z: " << U_here << " " << U(U_here) << " " << U(U_left_z) << "\n";
    
                        FZ(F_here) = -(kz/dh) * (U(U_here) - U(U_left_z));
                    }
        };

        auto fut_y = hpx::async(FluxY, lower, upper);
        auto fut_z = hpx::async(FluxZ, lower, upper);

        fut_y.get();
        fut_z.get();
    }

    IntVect ghostVect()
    {
        return config.ghost_vector;
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

