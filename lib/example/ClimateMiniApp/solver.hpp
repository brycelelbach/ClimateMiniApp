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

#include <boost/format.hpp>

#include "REAL.H"
#include "LevelData.H"
#include "FArrayBox.H"
#include "FluxBox.H"

#include "AsyncLevelData.H"
#include "AsyncExchange.H"
#include "StreamCSV.H"

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

namespace climate_mini_app
{

enum ProblemType
{
    Problem_Invalid             = -1,
    Problem_AdvectionDiffusion  = 1,
    Problem_Last                = Problem_AdvectionDiffusion
};

inline std::ostream& operator<<(std::ostream& os, ProblemType problem)
{ // {{{
    switch (problem)
    {
        default:
        case Problem_Invalid:
            os << "Invalid";
            break;
        case Problem_AdvectionDiffusion:
            os << "Advection-Diffusion";
            break;
    }
    return os;
} // }}}

struct configuration
{
    ///////////////////////////////////////////////////////////////////////////
    // Parameters.

    ProblemType const problem;

    std::uint64_t const nh; ///< "Horizontal" extent (y and z dimensions) per core.
    std::uint64_t const nv; ///< "Vertical" extent (x dimension) per core.

    std::uint64_t const max_box_size;

    IntVect const ghost_vector;

    bool const header;  ///< Print header for CSV timing data.
    bool const verbose; ///< Print status updates.

#if defined(CH_USE_HDF5)
    bool const output; ///< Generate HDF5 output.
#endif    

    ///////////////////////////////////////////////////////////////////////////

    configuration(
        ProblemType problem_
      , std::uint64_t nh_
      , std::uint64_t nv_
      , std::uint64_t max_box_size_
      , IntVect ghost_vector_
      , bool header_
      , bool verbose_
#if defined(CH_USE_HDF5)
      , bool output_
#endif
        )
      : problem(problem_)
      , nh(nh_)
      , nv(nv_)
      , max_box_size(max_box_size_)
      , ghost_vector(ghost_vector_)
      , header(header_)
      , verbose(verbose_)
#if defined(CH_USE_HDF5)
      , output(output_)
#endif
    {}

    std::string print_csv_header() const
    { // {{{
        return "Problem,"
               "Horizontal Extent (nh),"
               "Vertical Extent (nv),"
               "Max Box Size (mbs)";
    } // }}}

    CSVTuple<ProblemType, std::uint64_t, std::uint64_t, std::uint64_t>
    print_csv() const
    { // {{{
        return StreamCSV(problem, nh, nv, max_box_size); 
    } // }}}
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
{ // {{{
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
} // }}}

bool lower_boundary(boundary_type type)
{ // {{{
    return !upper_boundary(type);
} // }}}

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
    { // {{{
        define(dbl, comps, ghost);
    } // }}}

    void define(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    { // {{{
        U.define(dbl, comps, ghost);

        auto defineFY =
            [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
            {
                Box b = ld.disjointBoxLayout()[di];
                b.grow(ld.ghostVect());
                b = surroundingNodes(b, 1);
                ld[di].define(b, ld.nComp());
            }; 

        auto defineFZ =
            [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
            {
                Box b = ld.disjointBoxLayout()[di];
                b.grow(ld.ghostVect());
                b = surroundingNodes(b, 2);
                ld[di].define(b, ld.nComp());
            }; 
 
        FY.define(dbl, comps, ghost, defineFY);
        FZ.define(dbl, comps, ghost, defineFZ);

        epoch_.define(dbl);

        DataIterator dit(U.dataIterator());

        for (dit.begin(); dit.ok(); ++dit)
        {
            epoch_[dit()] = 0;
        }
    } // }}}

    hpx::future<void> exchangeAllAsync()
    { // {{{
        DataIterator dit = U.dataIterator();

        std::vector<hpx::future<void> > exchanges;
    
        for (dit.begin(); dit.ok(); ++dit)
            exchanges.push_back(LocalExchangeAsync(epoch_[dit()]++, dit(), U));

        return hpx::lcos::when_all(exchanges);
    } // }}}

    void exchangeAllSync()
    { // {{{
        exchangeAllAsync().get();
    } // }}}

    AsyncLevelData<FArrayBox> U;
    AsyncLevelData<FArrayBox> FY;
    AsyncLevelData<FArrayBox> FZ;

  private:
    LayoutData<std::size_t> epoch_;

  public:
    friend struct sub_problem_state;
};

struct sub_problem_state
{
    sub_problem_state()
      : ps_(0)
      , di_()
    {}

    sub_problem_state(problem_state& ps, DataIndex di)
      : ps_(&ps)
      , di_(di)
    {}

    sub_problem_state(sub_problem_state const& other)
      : ps_(other.ps_)
      , di_(other.di_)
/*
      , U(ps_->U[di_].interval(), ps_->U[di_])
      , FY(ps_->FY[di_].interval(), ps_->FY[di_])
      , FZ(ps_->FZ[di_].interval(), ps_->FZ[di_])
*/
    {}

    sub_problem_state& operator=(sub_problem_state const& other)
    { // {{{
        ps_ = other.ps_;
        di_ = other.di_;
/*
        U.define(ps_->U[di_].interval(), ps_->U[di_]);
        FY.define(ps_->FY[di_].interval(), ps_->FY[di_]);
        FZ.define(ps_->FZ[di_].interval(), ps_->FZ[di_]);
*/
        return *this;
    } // }}}

    void copy(sub_problem_state const& A)
    { // {{{
        U().copy(A.U());
        FY().copy(A.FY());
        FZ().copy(A.FZ());
    } // }}}

    void setVal(Real val)
    { // {{{
        U().setVal(val);
        FY().setVal(val);
        FZ().setVal(val);
    } // }}}

    void plus(sub_problem_state const& A, Real factor = 1.0)
    { // {{{
        U().plus(A.U(), factor);
    } // }}}

    hpx::future<void> exchangeAsync()
    { // {{{
        assert(ps_);
        return LocalExchangeAsync(ps_->epoch_[di_]++, di_, ps_->U);
    } // }}}

    void exchangeSync()
    { // {{{
        assert(ps_);
        LocalExchangeSync(ps_->epoch_[di_]++, di_, ps_->U);
    } // }}}

    FArrayBox& U()
    { // {{{
        assert(ps_);
        return ps_->U[di_]; 
    } // }}}

    FArrayBox const& U() const
    { // {{{
        assert(ps_);
        return ps_->U[di_]; 
    } // }}}

    FArrayBox& FY()
    { // {{{
        assert(ps_);
        return ps_->FY[di_]; 
    } // }}}

    FArrayBox const& FY() const
    { // {{{
        assert(ps_);
        return ps_->FY[di_]; 
    } // }}}

    FArrayBox& FZ()
    { // {{{
        assert(ps_);
        return ps_->FZ[di_]; 
    } // }}}

    FArrayBox const& FZ() const
    { // {{{
        assert(ps_);
        return ps_->FZ[di_]; 
    } // }}}

  private:
    problem_state* ps_;
    DataIndex di_;
};

template <typename Profile>
struct imex_operators
{
    imex_operators(Profile const& profile_)
      : profile(profile_)
    {}

    // TODO: Implement caching in profile.
    void resetDt(Real)
    {}

    void horizontalFlux(Real t, sub_problem_state& phi_) const
    { // {{{
        auto const& phi = phi_.U();

        auto& FY = phi_.FY();
        auto& FZ = phi_.FZ(); 

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.ghostVect());
        upper.shift(-1*profile.ghostVect());

        auto FluxY = [&](IntVect lower, IntVect upper)
        {
            for (auto k = lower[2]; k <= upper[2]; ++k)
                for (auto j = phi.smallEnd()[1]; j <= phi.bigEnd()[1]; ++j)
                    for (auto i = lower[0]; i <= upper[0]; ++i)
                    {
                        IntVect here(i, j, k);

                        FY(here) = profile.horizontal_flux_stencil(here, 1, phi);
                    };
        };

        auto FluxZ = [&](IntVect lower, IntVect upper)
        {
            for (auto k = phi.smallEnd()[2]; k <= phi.bigEnd()[2]; ++k)
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

    void explicitOp(
        Real t
      , std::size_t stage
      , sub_problem_state& kE_
      , sub_problem_state& phi_
        ) 
    { // {{{
        phi_.exchangeSync();
        horizontalFlux(t, phi_);

        auto&       kE  = kE_.U();
        auto const& phi = phi_.U();
        auto const& FY  = phi_.FY();
        auto const& FZ  = phi_.FZ();
 
        IntVect lower, upper;

        std::tie(lower, upper) = profile.boundary_conditions(t, kE_, phi_); 
    
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
    } // }}}

    void implicitOp(
        Real t
      , std::size_t stage
      , sub_problem_state& kI_
      , sub_problem_state& phi_
        )
    { // {{{
        kI_.setVal(0.0);
    } // }}}

    void solve(
        Real t
      , std::size_t stage
      , Real dtscale
      , sub_problem_state& phi_
        )
    { // {{{
        phi_.exchangeSync();

        auto& phi = phi_.U();
 
        IntVect lower, upper;

        std::tie(lower, upper) = profile.boundary_conditions(t, phi_, phi_); 

        Box b(lower, upper);

        // FIXME: Was this made serial for performance reasons?
        for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k)
            {
                auto A = profile.vertical_operator(j, k, phi, dtscale, b);

                profile.vertical_solve(j, k, A, phi, b);
            }
    } // }}}

  private:
    Profile profile;
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

    // FIXME: This should stop being a tuple
    std::tuple<Real, Real, Real> face_coords(IntVect here) const
    { // {{{
        return std::tuple<Real, Real, Real>(
            Real(here[0])/(Real(config.nv))
          , Real(here[1])/(Real(config.nh))
          , Real(here[2])/(Real(config.nh))
        );
    } // }}}

    std::tuple<Real, Real, Real> center_coords(IntVect here) const
    { // {{{
        return std::tuple<Real, Real, Real>(
            std::get<0>(face_coords(here))+0.5*std::get<0>(dp()) 
          , std::get<1>(face_coords(here))+0.5*std::get<1>(dp()) 
          , std::get<2>(face_coords(here))+0.5*std::get<2>(dp()) 
        );
    } // }}}

    // Spatial step size
    std::tuple<Real, Real, Real> dp() const
    { // {{{
        return face_coords(IntVect(1, 1, 1));
    } // }}}

    bool is_outside_domain(boundary_type bdry, int l) const
    { // {{{
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
                return (l <= -1);
            case UPPER_X:
                return (l >= config.nv);

            // Horizontal
            case LOWER_Y:
            case LOWER_Z:
                return (l <= -1); 
            case UPPER_Y:
            case UPPER_Z:
                return (l >= config.nh);
        };

        assert(false);
        return false;
    } // }}} 

    bool is_outside_domain(IntVect here) const
    { // {{{
        return is_outside_domain(LOWER_X, here[0])
            || is_outside_domain(LOWER_Y, here[1])
            || is_outside_domain(LOWER_Z, here[2])
            || is_outside_domain(UPPER_X, here[0])
            || is_outside_domain(UPPER_Y, here[1])
            || is_outside_domain(UPPER_Z, here[2]);
    } // }}}

    bool is_outside_domain(Real x, Real y, Real z) const
    { // {{{
        return is_outside_domain(LOWER_X, x)
            || is_outside_domain(LOWER_Y, y)
            || is_outside_domain(LOWER_Z, z)
            || is_outside_domain(UPPER_X, x)
            || is_outside_domain(UPPER_Y, y)
            || is_outside_domain(UPPER_Z, z);
    } // }}}

    bool is_boundary(boundary_type bdry, int l) const
    { // {{{
        switch (bdry)
        {
            // Vertical
            case LOWER_X:
                return (l == 0);
            case UPPER_X:
                return (l == config.nv-1);

            // Horizontal
            case LOWER_Y:
            case LOWER_Z:
                return (l == 0);
            case UPPER_Y:
            case UPPER_Z:
                return (l == config.nh-1);
        };

        assert(false);
        return false; 
    } // }}}

    bool is_boundary(IntVect here) const
    { // {{{
        return is_boundary(LOWER_X, here[0])
            || is_boundary(LOWER_Y, here[1])
            || is_boundary(LOWER_Z, here[2])
            || is_boundary(UPPER_X, here[0])
            || is_boundary(UPPER_Y, here[1])
            || is_boundary(UPPER_Z, here[2]);
    } // }}}

    bool is_boundary(Real x, Real y, Real z) const
    { // {{{
        return is_boundary(LOWER_X, x)
            || is_boundary(LOWER_Y, y)
            || is_boundary(LOWER_Z, z)
            || is_boundary(UPPER_X, x)
            || is_boundary(UPPER_Y, y)
            || is_boundary(UPPER_Z, z);
    } // }}}

    configuration const config;

  private:
    Derived const& derived() const
    { // {{{
        return static_cast<Derived const&>(*this);
    } // }}}
};

struct advection_diffusion_profile : profile_base<advection_diffusion_profile>
{
    typedef profile_base<advection_diffusion_profile> base_type;

    advection_diffusion_profile(
        configuration config_
      , Real cx_, Real cy_, Real cz_
      , Real kx_, Real vy_, Real vz_
        )
      : base_type(config_)
      , cx(cx_), cy(cy_), cz(cz_)
      , kx(kx_), vy(vy_), vz(vz_)
    {}

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
            (config.nv*numProc()-1),
            (config.nh*numProc()-1),
            (config.nh*numProc()-1)
        );

        bool is_periodic[] = { true, true, true };
        ProblemDomain domain(lower_bound, upper_bound, is_periodic);

        return domain;
    } // }}} 

    std::tuple<IntVect, IntVect> boundary_conditions(
        Real t
      , sub_problem_state& O_
      , sub_problem_state const& phi_
        ) const
    { // {{{
        auto const& phi = phi_.U();

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd();

        lower.shift(ghostVect());
        upper.shift(-1*ghostVect());

        return std::make_pair(lower, upper);
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

        // This is why we've picked 'x' as our vertical dimension; x columns
        // are contiguous in memory.

        assert((&phi(IntVect(lower[0]+1, j, k)) - &phi(IntVect(lower[0], j, k))) == 1);

        Real* rhs = &phi(IntVect(lower[0], j, k)); 

//        for (int i = lower[0]; i <= upper[0]; ++i)
//            std::cout << "phi[" << i << "] == " << rhs[i] << "\n";

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

//        for (int i = lower[0]; i <= upper[0]; ++i)
//            std::cout << "rhs[" << i << "] == " << rhs[i] << "\n";
 
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

