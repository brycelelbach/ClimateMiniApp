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
    template <std::size_t Dir>
    void horizontalFluxDirectional(
        FArrayBox& F
      , FArrayBox const& phi
      , IntVect lower
      , IntVect upper
        ) const
    { // {{{
        for (std::size_t d = 0; d < 3; ++d)
        {
            if (Dir == d) continue;

            lower.shift(d, profile.ghostVect()[d]);
            upper.shift(d, -1*profile.ghostVect()[d]);
        }

        //#pragma omp parallel for schedule(dynamic)
        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                {
                    IntVect here(i, j, k);

                    F(here) = profile.horizontal_flux_stencil<Dir>(here, phi);
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

        horizontalFluxDirectional<1>(FY, phi, lower, upper);
        horizontalFluxDirectional<2>(FZ, phi, lower, upper);
    } // }}}

    template <typename X>
    void explicitOp(
        Real t
      , std::size_t stage
      , X&& kE_
      , X&& phi_
        ) const
    { // {{{
        if (explicitOpDisabled())
            return;

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

        //#pragma omp parallel for schedule(dynamic)
        for (auto i = lower[0]; i <= upper[0]; ++i)
            for (auto j = lower[1]; j <= upper[1]; ++j)
            {
                auto k = lower[2];

                Real const inv_dh_c = profile.inverse_dh_c();

                for (; k <= upper[2]; k += 4)
                {
                    IntVect h0(i, j, k  );
                    IntVect h1(i, j, k+1);
                    IntVect h2(i, j, k+2);
                    IntVect h3(i, j, k+3);

                    Real ke0, ke1, ke2, ke3;

                    ke0 = profile.horizontal_stencil(h0, FY, FZ, inv_dh_c);
                    ke1 = profile.horizontal_stencil(h1, FY, FZ, inv_dh_c);
                    ke2 = profile.horizontal_stencil(h2, FY, FZ, inv_dh_c);
                    ke3 = profile.horizontal_stencil(h3, FY, FZ, inv_dh_c);

                    kE(h0) = ke0; 
                    kE(h1) = ke1; 
                    kE(h2) = ke2; 
                    kE(h3) = ke3; 
                }

                // Handle the remainder, if there is one.
                for (; k <= upper[2]; ++k)
                {
                    IntVect h0(i, j, k  );

                    kE(h0) = profile.horizontal_stencil(h0, FY, FZ, inv_dh_c);
                }
            }
    } // }}}

    template <typename X>
    void implicitOp(
        Real t
      , std::size_t stage
      , X&& kI_
      , X&& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
            return;

        auto& kI  = kI_.U();
        auto& phi = phi_.U();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        //#pragma omp parallel for schedule(dynamic)
        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto k = lower[2]; k <= upper[2]; ++k)
            {
                LapackFactorization A;

                profile.build_vertical_operator_for_apply(A, b);

                profile.apply_vertical_operator(j, k, A, b, kI, phi);
            }
    } // }}}

    template <typename X>
    void solve(
        Real t
      , std::size_t stage
      , Real dtscale
      , X&& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
            return;

        // For HPX, we operate on sub_problem_states (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        #if defined(CH_HPX)
            phi_.exchangeSync();
        #endif

        auto& phi = phi_.U();

        auto& factor_flops = phi_.factor_flops();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        //#pragma omp parallel for schedule(dynamic)
        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto k = lower[2]; k <= upper[2]; ++k)
            {
                LapackFactorization A;

                profile.build_vertical_operator_for_solve(A, b, dtscale);

                profile.vertical_solve(j, k, A, b, phi);
            }
    } // }}}

    bool implicitOpDisabled() const
    { // {{{
        return profile.implicitOpDisabled();
    } // }}}

    bool explicitOpDisabled() const
    { // {{{
        return profile.explicitOpDisabled();
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
/*
        #if !defined(CH_HPX)
            phi_.exchangeSync();
        #endif
*/

        DataIterator dit = phi_.U.dataIterator();
        std::size_t const nbox = dit.size();

        //#pragma omp parallel for schedule(dynamic)
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
        std::size_t const nbox = dit.size();

        //#pragma omp parallel for schedule(dynamic)
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
/*
        #if !defined(CH_HPX)
            phi_.exchangeSync();
        #endif
*/

        DataIterator dit = phi_.U.dataIterator();
        std::size_t const nbox = dit.size();

        //#pragma omp parallel for schedule(dynamic)
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

