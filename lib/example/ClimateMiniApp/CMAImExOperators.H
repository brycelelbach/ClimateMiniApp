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

#include "Lapack.H"
#include "LapackFactorization.H"
#include "LapackWrapper.H"

namespace climate_mini_app
{

template <typename Profile>
struct imex_per_box_operators
{
    imex_per_box_operators(Profile const& profile_)
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

        // Compute the two horizontal fluxes in parallel if using HPX.
        #if defined(CH_HPX)
            using std::ref;
            using std::cref;
   
            auto F = &imex_per_box_operators::horizontalFluxDirectional;
 
            auto futY = hpx::async(F, this, ref(FY), cref(phi), 1, lower, upper);
            auto futZ = hpx::async(F, this, ref(FZ), cref(phi), 2, lower, upper);
    
            futY.get();
            futZ.get();
        #else
            horizontalFluxDirectional(FY, phi, 1, lower, upper);
            horizontalFluxDirectional(FZ, phi, 2, lower, upper);
        #endif
    } // }}}

    void explicitOp(
        Real t
      , std::size_t stage
      , sub_problem_state& kE_
      , sub_problem_state& phi_
        ) const
    { // {{{
        // For HPX, we operate on sub_problem_states (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        #if defined(CH_HPX)
            phi_.exchangeSync();
        #endif

        horizontalFlux(phi_);

        auto&       kE  = kE_.U();
        auto const& phi = phi_.U();
        auto const& FY  = phi_.FY();
        auto const& FZ  = phi_.FZ();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

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

        auto& kI  = kI_.U();
        auto& phi = phi_.U();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto k = lower[2]; k <= upper[2]; ++k)
            {
                LapackFactorization A;

                profile.build_vertical_operator_for_apply(A, b);
                profile.apply_vertical_operator(j, k, A, b, kI, phi);
            }
    } // }}}

    void solve(
        Real t
      , std::size_t stage
      , Real dtscale
      , sub_problem_state& phi_
        ) const
    { // {{{
        // For HPX, we operate on sub_problem_states (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        #if defined(CH_HPX)
            phi_.exchangeSync();
        #endif

        auto& phi = phi_.U();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto k = lower[2]; k <= upper[2]; ++k)
            {
                LapackFactorization A;

                profile.build_vertical_operator_for_solve(A, b, dtscale);
                profile.vertical_solve(j, k, A, b, phi);
            }
    } // }}}

  private:
    Profile const profile;
};

template <typename Profile>
struct imex_per_ld_operators
{
    imex_per_ld_operators(Profile const& profile_)
    // {{{
      : profile(profile_)
    {} // }}}

    // TODO: Implement caching in profile.
    void resetDt(Real)
    { // {{{
    } // }}}

    void explicitOp(
        Real t
      , std::size_t stage
      , problem_state& kE_
      , problem_state& phi_
        ) const
    { // {{{
        // If we're using MPI, we do the exchange at problem_state (e.g.
        // LevelData) granularity, and the exchange at the sub_problem_state
        // level (e.g. FArrayBox granularity) is #ifdef'd out. 
        #if !defined(CH_HPX)
            phi_.exchangeSync();
        #endif

        DataIterator dit = phi_.U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_phi(phi_, dit[ibox]);
            sub_problem_state sub_kE(kE_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.explicitOp(t, stage, sub_kE, sub_phi);
        }
    } // }}}

    void implicitOp(
        Real t
      , std::size_t stage
      , problem_state& kI_
      , problem_state& phi_
        ) const
    { // {{{
        DataIterator dit = phi_.U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_phi(phi_, dit[ibox]);
            sub_problem_state sub_kI(kI_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.implicitOp(t, stage, sub_kI, sub_phi);
        }
    } // }}}

    void solve(
        Real t
      , std::size_t stage
      , Real dtscale
      , problem_state& phi_
        ) const
    { // {{{
        // If we're using MPI, we do the exchange at problem_state (e.g.
        // LevelData) granularity, and the exchange at the sub_problem_state
        // level (FArrayBox granularity) is #ifdef'd out. 
        #if !defined(CH_HPX)
            phi_.exchangeSync();
        #endif

        DataIterator dit = phi_.U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_phi(phi_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.solve(t, stage, dtscale, sub_phi);
        }
    } // }}}

  private:
    Profile const profile;
};

}

#endif // CHOMBO_9AA05D23_647F_48CA_ADE1_CB053BABE61E

