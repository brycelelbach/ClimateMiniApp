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

#if !defined(CHOMBO_9AA05D23_647F_48CA_ADE1_CB053BABE61E)
#define CHOMBO_9AA05D23_647F_48CA_ADE1_CB053BABE61E

#include "CMAProblemState.H"

namespace climate_mini_app
{

template <typename Profile>
struct imex_operators
{
    imex_operators(Profile const& profile_)
    // {{{
      : profile(profile_)
    {} // }}}

    // TODO: Implement caching in profile.
    void resetDt(Real)
    { // {{{
    } // }}}

  private:
    void horizontalFluxDirectional(
        FArrayBox& F
      , FArrayBox const& phi
      , std::size_t dir
      , IntVect lower
      , IntVect upper
        ) const
    { // {{{
        for (std::size_t d = 0; d < 3; ++d)
        {
            if (dir == d) continue;

            lower.shift(d, profile.ghostVect()[d]);
            upper.shift(d, -1*profile.ghostVect()[d]);
        }

        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                {
                    IntVect here(i, j, k);

                    F(here) = profile.horizontal_flux_stencil(here, dir, phi);
                };
    } // }}}

  public:
    void horizontalFlux(sub_problem_state& phi_) const
    { // {{{
        auto const& phi = phi_.U();

        auto& FY = phi_.FY();
        auto& FZ = phi_.FZ(); 

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        using std::ref;
        using std::cref;

        auto fut_y = hpx::async(&imex_operators::horizontalFluxDirectional
                              , this, ref(FY), cref(phi), 1, lower, upper);
        auto fut_z = hpx::async(&imex_operators::horizontalFluxDirectional
                              , this, ref(FZ), cref(phi), 2, lower, upper);

        fut_y.get();
        fut_z.get();
    } // }}}

    void explicitOp(
        Real t
      , std::size_t stage
      , sub_problem_state& kE_
      , sub_problem_state& phi_
        ) const
    { // {{{
        phi_.exchangeSync();
        horizontalFlux(phi_);

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
        ) const
    { // {{{
        kI_.setVal(0.0);
    } // }}}

    void solve(
        Real t
      , std::size_t stage
      , Real dtscale
      , sub_problem_state& phi_
        ) const
    { // {{{
        phi_.exchangeSync();
    } // }}}

  private:
    Profile const profile;
};

}

#endif // CHOMBO_9AA05D23_647F_48CA_ADE1_CB053BABE61E

