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

// FIXME: reflux_horizontal, horizontal_stencil and vertical_solve use
// inconsistent parameter names and local variables.

// Solving problems of the form:
//
//      U_t = (kx * U_xx + ky * U_yy + kz * U_zz)   [Diffusion]
//          -             (vy * U_y  + vz * U_z)    [Advection]
//          + h                                     [Sources] 

namespace transport_3d
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
      , FY()
      , FZ()
      , epoch_()
    {}

    problem_state(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
      : U()
      , FY()
      , FZ()
      , epoch_()
    {
        define(dbl, comps, ghost);
    }

    void define(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    {
        U.define(dbl, ghost);
        DefineData(U, comps);
 
        FY.define(dbl, ghost);
        FZ.define(dbl, ghost);

        epoch_.define(dbl);

        DataIterator dit(U.dataIterator());

        for (dit.begin(); dit.ok(); ++dit)
        {
            Box by = dbl[dit()];
            by.grow(ghost);
            by = surroundingNodes(by, 1);
            FY[dit()].define(by, comps);

            Box bz = dbl[dit()];
            bz.grow(ghost);
            bz = surroundingNodes(bz, 2);
            FZ[dit()].define(bz, comps);

            epoch_[dit()] = 0;
        }
    }

    void copy(DataIndex di, problem_state const& A)
    {
        U[di].copy(A.U[di]);
        FY[di].copy(A.FY[di]);
        FZ[di].copy(A.FZ[di]);
    }

    void zero(DataIndex di)
    {
        U[di].setVal(0.0);
        FY[di].setVal(0.0);
        FZ[di].setVal(0.0);
    }

    void increment(DataIndex di, problem_state const& A, Real factor = 1.0)
    {
        U[di].plus(A.U[di], factor);
    }

    void exchangeSync(DataIndex di)
    { 
        LocalExchangeSync(epoch_[di]++, di, U);
    }

    hpx::future<void> exchangeAllAsync()
    {
        DataIterator dit = U.dataIterator();

        std::vector<hpx::future<void> > exchanges;
    
        for (dit.begin(); dit.ok(); ++dit)
        {
            exchanges.push_back(
                hpx::async(
                    [&](DataIndex di) { this->exchangeSync(di); }
                  , dit()
                ) 
            );
        }

        return hpx::lcos::when_all(exchanges);
    }

    void exchangeAllSync()
    {
        exchangeAllAsync().get();
    }

    AsyncLevelData<FArrayBox> U;
    AsyncLevelData<FArrayBox> FY;
    AsyncLevelData<FArrayBox> FZ;

  private:
    LayoutData<std::size_t> epoch_;
};

template <typename Profile>
struct imex_operators
{
    imex_operators(Profile const& profile_)
      : profile(profile_)
    {}

    // TODO: Implement caching in profile.
    void resetDt(Real)
    {
    }

    void horizontalFlux(DataIndex di, Real t, problem_state& phi_) const
    { // {{{
        auto const& phi = phi_.U[di];

        auto& FY = phi_.FY[di];
        auto& FZ = phi_.FZ[di]; 

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.ghostVect());
        upper.shift(-1*profile.ghostVect());

        auto FluxY = [&](IntVect lower, IntVect upper)
        {
            for (auto k = lower[2]; k <= upper[2]; ++k)
                for (auto j = phi.smallEnd()[1]+1; j <= phi.bigEnd()[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        IntVect here(i, j, k);

                        FY(here) = profile.horizontal_flux_stencil(here, 1, phi);
                    };
        };

        auto FluxZ = [&](IntVect lower, IntVect upper)
        {
            for (auto k = phi.smallEnd()[2]+1; k <= phi.bigEnd()[2]; ++k)
                for (auto j = lower[1]; j <= upper[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        IntVect here(i, j, k);
    
                        FZ(here) = profile.horizontal_flux_stencil(here, 2, phi);
                    };
        };

        auto fut_y = hpx::async(FluxY, lower, upper);
        auto fut_z = hpx::async(FluxZ, lower, upper);

        fut_y.get();
        fut_z.get();
    } // }}}

    // TODO: Lift stencil.
    void explicitOp(
        DataIndex di
      , Real t
      , problem_state& kE_
      , problem_state& phi_
        ) 
    {
        phi_.exchangeSync(di);
        horizontalFlux(di, t, phi_);

        auto&       kE  = kE_.U[di];
        auto const& phi = phi_.U[di];
        auto const& FY  = phi_.FY[di];
        auto const& FZ  = phi_.FZ[di];
 
        IntVect lower, upper;

        std::tie(lower, upper) = profile.boundary_conditions(di, t, kE_, phi_); 

        lower.shift(profile.ghostVect());
        upper.shift(-1*profile.ghostVect());
    
        ///////////////////////////////////////////////////////////////////////
        // Interior points.
        for (auto i = lower[0]; i <= upper[0]; ++i)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto k = lower[2]; k <= upper[2]; ++k)
                {
                    IntVect here(i, j, k);

                    kE(here) = profile.horizontal_stencil(here, FY, FZ)
                             + profile.source_term(here, t); 
                }
    }

    void implicitOp(
        DataIndex di
      , Real t
      , problem_state& kI
      , problem_state& phi
        )
    {
        kI.zero(di);
    }

    void solve(
        DataIndex di
      , Real t
      , Real dtscale
      , problem_state& phi_
        )
    {
        phi_.exchangeSync(di);

        auto& phi = phi_.U[di];
 
        IntVect lower, upper;

        std::tie(lower, upper) = profile.boundary_conditions(di, t, phi_, phi_); 

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

template <typename Profile>
struct dirichlet_boundary_conditions
{
    dirichlet_boundary_conditions(
        Profile const& profile
      , DataIndex di
      , Real t
      , problem_state& O
      , problem_state& phi
        )
      : profile_(profile)
      , di_(di)
      , t_(t)
      , O_(O)
      , phi_(phi)
    {} 
    
    IntVect operator()(
        boundary_type type
      , std::size_t dir
      , IntVect V
        ) const
    { 
        auto&       O      = O_.U[di];
        auto const& phi    = phi_.U[di];

        int sign = (upper_boundary(type) ? -1 : 1);
 
        if (profile_.is_outside_domain(type, V[dir]))
        {
            std::size_t A = -1, B = -1;
    
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
    
                    for (int c = 0; c <= profile_.ghostVect()[dir]; ++c)
                    {
                        out.shift(dir, sign*c);
                        O(out) = profile_.outside_domain(type, out, phi, t); 
                    } 
                }
        }

        if (profile_.is_boundary(type, (V+sign*profile_.ghostVect())[dir]))
        {
            std::size_t A = -1, B = -1;
    
            if      (0 == dir) { A = 1; B = 2; }
            else if (1 == dir) { A = 0; B = 2; }
            else if (2 == dir) { A = 0; B = 1; }
            else    assert(false);

            for (int a = phi.smallEnd()[A]; a <= phi.bigEnd()[A]; ++a)
                for (int b = phi.smallEnd()[B]; b <= phi.bigEnd()[B]; ++b)
                {
                    IntVect bdry(V);
                    bdry.shift(dir, sign*profile_.ghostVect()[dir]);
                    bdry.setVal(A, a);
                    bdry.setVal(B, b);

                    O(bdry) = profile_.boundary(type, bdry, phi, t); 
                }

            int sign = (upper_boundary(type) ? -1 : 1);
            V.shift(dir, sign*IntVect::Unit[dir]);
        }

        return V;
    }

  private:
    Profile const& profile_;
    DataIndex di_;
    Real t_;
    problem_state& O_;
    problem_state& phi_;
}; 

template <typename Derived>
struct profile_base 
{
  public:
    profile_base(configuration config_)
      : config(config_)
    {}

    IntVect ghostVect() const
    { // {{{
        return config.ghost_vector;
    } // }}}

    // FIXME: Switch to cell centered
    std::tuple<Real, Real, Real> phys_coords(IntVect here) const
    { // {{{
        return std::tuple<Real, Real, Real>(
            Real(here[0])/(Real(config.nv)-1.0)
          , Real(here[1])/(Real(config.nh)-1.0)
          , Real(here[2])/(Real(config.nh)-1.0)
        );
    } // }}}

    // Spatial step size
    std::tuple<Real, Real, Real> dp() const
    { // {{{
        return phys_coords(IntVect(1, 1, 1));
    } // }}}

    bool is_outside_domain(boundary_type bdry, int l) const
    { // {{{
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
    } // }}} 

    bool is_outside_domain(IntVect here) const
    { // {{{
        return is_outside_domain(LOWER_X, here[0])
            || is_outside_domain(LOWER_Y, here[1])
            || is_outside_domain(LOWER_Z, here[2]);
    } // }}}

    bool is_outside_domain(Real x, Real y, Real z) const
    { // {{{
        return is_outside_domain(LOWER_X, x)
            || is_outside_domain(LOWER_Y, y)
            || is_outside_domain(LOWER_Z, z);
    } // }}}

    bool is_boundary(boundary_type bdry, int l) const
    { // {{{
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
    } // }}}

    bool is_boundary(IntVect here) const
    { // {{{
        return is_boundary(LOWER_X, here[0])
            || is_boundary(LOWER_Y, here[1])
            || is_boundary(LOWER_Z, here[2]);
    } // }}}

    bool is_boundary(Real x, Real y, Real z) const
    { // {{{
        return is_boundary(LOWER_X, x)
            || is_boundary(LOWER_Y, y)
            || is_boundary(LOWER_Z, z);
    } // }}}

  private:
    Derived const& derived() const
    {
        return static_cast<Derived const&>(*this);
    }

  protected:
    configuration config;

};

struct transport_profile : profile_base<transport_profile>
{
    typedef profile_base<transport_profile> base_type;

    transport_profile(
        configuration config_
      , Real A_, Real B_, Real C_
      , Real kx_, Real ky_, Real kz_
      , Real vy_, Real vz_
        )
      : base_type(config_)
      , A(A_), B(B_), C(C_)
      , kx(kx_), ky(ky_), kz(kz_)
      , vy(vy_), vz(vz_)
    {}

    // Time step size
    Real dt() const
    { // {{{
        // For advection-diffusion: 
        //
        //      dt = CFL*min( (dh*dh)/(2*ky)
        //                  , (dh*dh)/(2*kz)
        //                  , (2*ky)/(vy*vy)
        //                  , (2*kz)/(vz*vz)
        //                  )
        //
        // If we just have advection:
        //
        //      dt = CFL*min( dh/vy
        //                  , dh/vz
        //                  )
        //
        Real constexpr CFL = 0.9;
        Real const dh = std::get<1>(dp()); 

        // (ky>0) || (kz>0) || (vy!=0) || (vz!=0)
        assert(  (ky > 0.0)
              || (kz > 0.0)
              || (std::fabs(vy-0.0) > 1e-16)
              || (std::fabs(vz-0.0) > 1e-16));

        double dt = 10000.0;

        if ((ky > 0.0) || (kz > 0.0))
            dt = std::min(dt, (dh*dh)/(2.0*std::max(ky, kz)));
        else
            dt = std::min(dt, dh/std::max(vy, vz));

        if ((ky > 0.0) && (std::fabs(vy-0.0) > 1e-16))
            dt = std::min(dt, (2.0*ky)/(vy*vy));

        if ((kz > 0.0) && (std::fabs(vz-0.0) > 1e-16))
            dt = std::min(dt, (2.0*kz)/(vz*vz));

        assert(std::fabs(dt-10000.0) < 1e-16);

        return CFL*dt;
    } // }}}

    ProblemDomain problem_domain() const
    {
        IntVect lower_bound(IntVect::Zero);
        IntVect upper_bound(
            (config.nv*numProc()-1),
            (config.nh*numProc()-1),
            (config.nh*numProc()-1)
        );

        bool is_periodic[] = { true, true, true };
        ProblemDomain domain(lower_bound, upper_bound, is_periodic);

        return domain;
    }    

    std::tuple<IntVect, IntVect> boundary_conditions(
        DataIndex di
      , Real t
      , problem_state& kE_
      , problem_state& phi_
        ) const
    { // {{{
        auto const& phi    = phi_.U[di];

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();

/*
        lower = BCs(LOWER_X, 0, lower);
        upper = BCs(UPPER_X, 0, upper);
        lower = BCs(LOWER_Y, 1, lower);
        upper = BCs(UPPER_Y, 1, upper);
        lower = BCs(LOWER_Z, 2, lower);
        upper = BCs(UPPER_Z, 2, upper);
*/

        return std::make_pair(lower, upper);
    } // }}}

/*
    Real boundary(
        boundary_type bdry
      , IntVect here
      , FArrayBox const& phi
      , Real t
        ) const
    {
        return 0.0;
    } 

    Real outside_domain(
        boundary_type bdry
      , IntVect here
      , FArrayBox const& phi
      , Real t
        ) const
    {
        return 0.0;
    } 
*/
    
    Real horizontal_flux_stencil(
        IntVect here
      , std::size_t dir
      , FArrayBox const& phi
        ) const
    { // {{{
        Real const dh = std::get<1>(dp()); 

        Real k = 0.0;
        Real v = 0.0;
        IntVect left;

        if      (1 == dir)
        { 
            k = ky;
            v = vy;
            left = IntVect(here[0], here[1]-1, here[2]);
        }

        else if (2 == dir)
        {
            k = kz;
            v = vz;
            left = IntVect(here[0], here[1], here[2]-1);
        }

        else
            assert(false);

        // NOTE: This currently assumes constant-coefficients 
        // and assumes we're just taking an average of the
        // coefficients. For variable-coefficient we need
        // to actually take an average. 
        // 
        // F_{j-1/2} ~= -k_{j-1/2}/h (phi_j - phi_{j-1})

        return -1.0 * ( (k/dh) * (phi(here) - phi(left))  // Diffusion 
                      - 0.5 * v * (phi(here) + phi(left)) // Advection 
                      );
    } // }}}

    // FIXME: IntVect indexing may be expensive, compare to Hans' loops
    Real horizontal_stencil(
        IntVect here
      , FArrayBox const& FY
      , FArrayBox const& FZ
        ) const
    { // {{{ 
        IntVect n(here[0], here[1], here[2]+1); // north
        IntVect s(here[0], here[1], here[2]);   // south
        IntVect e(here[0], here[1]+1, here[2]); // east
        IntVect w(here[0], here[1], here[2]);   // west

        Real const dh = std::get<1>(dp()); 
        return (-1.0/dh) * (FY(e) - FY(w) + FZ(n) - FZ(s)); 
    } // }}}

    typedef std::tuple<std::vector<Real>, std::vector<Real>, std::vector<Real> >
        crs_matrix;
    
    // TODO: Could be cached.
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
        Real const kvdv = kx/(dv*dv); 
        Real const H = dt()*dtscale;

        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(size-1, H*(kvdv/2.0));
        // Diagonal part of the matrix.
        std::vector<Real> d(size, 1.0-H*kvdv);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(size-1, H*(kvdv/2.0));

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
    } // }}}

    Real initial_state(IntVect here) const
    { // {{{
        return this->analytic_solution(here, 0.0);
    } // }}}

    Real analytic_solution(IntVect here, Real t) const
    { // {{{
        Real x, y, z;
        x = std::get<0>(phys_coords(here));
        y = std::get<1>(phys_coords(here));
        z = std::get<2>(phys_coords(here));

        //if (is_outside_domain(here)) return 0.0;
        //else if (is_boundary(here)) return 0.0;
        Real const C = 1.0;
        Real const c1 = 0.5;
        Real const c2 = 0.25;
        return (C/(4.0*t+c1))
             * std::exp(
                    -1.0*(((x-c2)*(x-c2))/(kx*(4.0*t+c1)))
                    - (((y-vy*t-c2)*(y-vy*t-c2))/(ky*(4.0*t+c1)))
                    - (((z-vz*t-c2)*(z-vz*t-c2))/(kz*(4.0*t+c1)))
               );
    } // }}}

    Real source_term(IntVect here, Real t) const
    { // {{{
        return 0.0; 
    } // }}}

  private:
    Real A;
    Real B;
    Real C;

  public:
    Real kx;
    Real ky;
    Real kz;

    Real vy;
    Real vz;
};

}

#endif // CHOMBO_4EFB6852_A674_4B87_80B6_D5FBE8086AF2

