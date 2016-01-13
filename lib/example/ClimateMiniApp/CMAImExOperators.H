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

//#include <advisor-annotate.h>

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
/*
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
            if (Dir == d)
            {
                lower.shift(d, 1);
                continue;
            } 

            lower.shift(d, profile.ghostVect()[d]);
            upper.shift(d, -1*profile.ghostVect()[d]);
        }

        //#pragma omp parallel for schedule(static)
        for (auto k = lower[2]; k <= upper[2]; ++k)
            for (auto j = lower[1]; j <= upper[1]; ++j)
                for (auto i = lower[0]; i <= upper[0]; ++i)
                {
                    IntVect here(i, j, k);

                    F(here) = profile.horizontal_flux_stencil<Dir>(here, phi);
                };
    } // }}}
*/

  public:
/*
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
*/

    template <std::size_t Stage, typename X, std::size_t TotalStages>
    void explicitOp(
        Real t
      , Real dt
      , Real const s_aE[TotalStages][TotalStages]
      , Real const s_b[TotalStages]
      , X& state_
      , X& kE_
      , std::array<X, TotalStages>& phi_
        ) const
    { // {{{
        if (explicitOpDisabled())
        {
            kE_.U().setVal(0.0);
            return;
        }

        // For HPX, we operate on sub_problem_states (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        #if defined(CH_HPX)
            phi_[Stage].exchangeSync();
        #endif

        //horizontalFlux(phi_);

        auto& state = state_.U();
        auto& kE    = kE_.U();
        auto& phi   = phi_[Stage].U();
        auto& FY    = phi_[Stage].FY();
        auto& FZ    = phi_[Stage].FZ();

        IntVect const lower_kE  = kE.smallEnd();  // FIXME: Better names.
        IntVect const upper_kE  = kE.bigEnd();    // FIXME: Better names.

        IntVect const lower_phi = phi.smallEnd(); // FIXME: Better names.
        IntVect const upper_phi = phi.bigEnd();   // FIXME: Better names.
 
        IntVect const lower_FY  = FY.smallEnd();  // FIXME: Better names.
        IntVect const upper_FY  = FY.bigEnd();    // FIXME: Better names.

        IntVect const lower_FZ  = FZ.smallEnd();  // FIXME: Better names.
        IntVect const upper_FZ  = FZ.bigEnd();    // FIXME: Better names.

        IntVect phys_lower_phi = phi.smallEnd();
        IntVect phys_upper_phi = phi.bigEnd(); 

        phys_lower_phi.shift(profile.config.ghost_vector);
        phys_upper_phi.shift(-1*profile.config.ghost_vector);

        IntVect phys_lower_FY = FY.smallEnd();
        IntVect phys_upper_FY = FY.bigEnd(); 

        phys_lower_FY.shift(0, profile.ghostVect()[0]);
        phys_upper_FY.shift(0, -1*profile.ghostVect()[0]);
        phys_lower_FY.shift(1, 1);
//        phys_lower_FY.shift(1, profile.ghostVect()[1]);
//        phys_upper_FY.shift(1, -1*profile.ghostVect()[1]);
        phys_lower_FY.shift(2, profile.ghostVect()[2]);
        phys_upper_FY.shift(2, -1*profile.ghostVect()[2]);

        IntVect phys_lower_FZ = FZ.smallEnd();
        IntVect phys_upper_FZ = FZ.bigEnd(); 

        phys_lower_FZ.shift(0, profile.ghostVect()[0]);
        phys_upper_FZ.shift(0, -1*profile.ghostVect()[0]);
        phys_lower_FZ.shift(1, profile.ghostVect()[1]);
        phys_upper_FZ.shift(1, -1*profile.ghostVect()[1]);
        phys_lower_FZ.shift(2, 1);
//        phys_lower_FZ.shift(2, profile.ghostVect()[2]);
//        phys_upper_FZ.shift(2, -1*profile.ghostVect()[2]);

        ///////////////////////////////////////////////////////////////////////

        Real* __restrict__ pstate = state.dataPtr(0); 
        Real* __restrict__ pke    = kE.dataPtr(0); 
        Real* __restrict__ pfy    = FY.dataPtr(0);
        Real* __restrict__ pfz    = FZ.dataPtr(0);

        std::array<Real* __restrict__, TotalStages> pphi;

        for (std::size_t k = 0; k < TotalStages; ++k)
            pphi[k] = phi_[k].U().dataPtr(0);

        Real const inv_dh_c = profile.inverse_dh_c();

        // len0 is the extent of i and the stride of j.
        // len1 is the extent of j. len0*len1 is the stride of k.
        auto const len0_kE  = (upper_kE[0]-lower_kE[0]+1);
        auto const len1_kE  = (upper_kE[1]-lower_kE[1]+1);

        auto const len0_phi = (upper_phi[0]-lower_phi[0]+1);
        auto const len1_phi = (upper_phi[1]-lower_phi[1]+1);

        auto const len0_FY  = (upper_FY[0]-lower_FY[0]+1);
        auto const len1_FY  = (upper_FY[1]-lower_FY[1]+1);

        auto const len0_FZ  = (upper_FZ[0]-lower_FZ[0]+1);
        auto const len1_FZ  = (upper_FZ[1]-lower_FZ[1]+1);

        for (auto k = phys_lower_FY[2]; k <= phys_upper_FY[2]; ++k)
        {
            auto const dif2_phi = k-lower_phi[2];
            auto const dif2_FY  = k-lower_FY[2];
            auto const dif2_FZ  = k-lower_FZ[2];

            for (auto j = phys_lower_FY[1]; j <= phys_upper_FY[1]; ++j)
            {
                auto const dif1_phi = j-lower_phi[1];
                auto const dif1_FY  = j-lower_FY[1];
                auto const dif1_FZ  = j-lower_FZ[1];

                #pragma simd
                #pragma ivdep
                for (auto i = phys_lower_FY[0]; i <= phys_upper_FY[0]; ++i)
                {
                    auto const idx_phi =
                        i-lower_phi[0] + len0_phi*(dif1_phi + len1_phi*dif2_phi);
                    auto const idx_FY =
                        i-lower_FY[0] + len0_FY*(dif1_FY + len1_FY*dif2_FY);

                    pfy[idx_FY] = profile.horizontal_flux_stencil<1>
                        (pphi[Stage], idx_phi, len0_phi);
                }
            }
        }

        for (auto k = phys_lower_FZ[2]; k <= phys_upper_FZ[2]; ++k)
        {
            auto const dif2_phi = k-lower_phi[2];
            auto const dif2_FY  = k-lower_FY[2];
            auto const dif2_FZ  = k-lower_FZ[2];

            for (auto j = phys_lower_FZ[1]; j <= phys_upper_FZ[1]; ++j)
            {
                auto const dif1_phi = j-lower_phi[1];
                auto const dif1_FY  = j-lower_FY[1];
                auto const dif1_FZ  = j-lower_FZ[1];

                #pragma simd
                #pragma ivdep
                for (auto i = phys_lower_FZ[0]; i <= phys_upper_FZ[0]; ++i)
                {
                    auto const idx_phi =
                        i-lower_phi[0] + len0_phi*(dif1_phi + len1_phi*dif2_phi);
                    auto const idx_FZ =
                        i-lower_FZ[0] + len0_FZ*(dif1_FZ + len1_FZ*dif2_FZ);

                    pfz[idx_FZ] = profile.horizontal_flux_stencil<2>
                        (pphi[Stage], idx_phi, len0_phi*len1_phi);
                }
            }
        }

//        #if defined(CH_HPX)
//            phi_[Stage].exchangeFluxSync();
//        #endif

        auto constexpr CX = 16;
        auto constexpr CY = 16;
        auto constexpr CZ = 16;

//        for (auto bk = phys_lower_phi[2]; bk <= phys_upper_phi[2]; bk+=CZ)
//        for (auto bj = phys_lower_phi[1]; bj <= phys_upper_phi[1]; bj+=CY)
//        for (auto bi = phys_lower_phi[0]; bi <= phys_upper_phi[0]; bi+=CX)

/*
        for ( auto k = bk
            ; k <= std::min(bk + CZ, phys_upper_phi[2])
            ; ++k
            )
*/
        for (auto k = phys_lower_phi[2]; k <= phys_upper_phi[2]; ++k)
        {
//            auto const dif2_kE = k-lower_kE[2];
            auto const dif2_phi = k-lower_phi[2];
            auto const dif2_FY  = k-lower_FY[2];
            auto const dif2_FZ  = k-lower_FZ[2];

/*
            for ( auto j = bj 
                ; j <= std::min(bj + CY, phys_upper_phi[1])
                ; ++j
                )
*/
            for (auto j = phys_lower_phi[1]; j <= phys_upper_phi[1]; ++j)
            {
//                auto const dif1_kE = j-lower_kE[1];
                auto const dif1_phi = j-lower_phi[1];
                auto const dif1_FY  = j-lower_FY[1];
                auto const dif1_FZ  = j-lower_FZ[1];

                #pragma simd
                #pragma ivdep
/*
                for ( auto i = bi
                    ; i <= std::min(bi + CX, phys_upper_phi[0])
                    ; ++i
                    )
*/
                for (auto i = phys_lower_phi[0]; i <= phys_upper_phi[0]; ++i)
                {
//                    auto const idx_kE =
//                        i-lower_kE[0] + len0_kE*(dif1_kE + len1_kE*dif2_kE);
                    auto const idx_phi =
                        i-lower_phi[0] + len0_phi*(dif1_phi + len1_phi*dif2_phi);
                    auto const idx_FY =
                        i-lower_FY[0] + len0_FY*(dif1_FY + len1_FY*dif2_FY);

                    Real kE_here = profile.horizontal_stencil_1d<1>
                        (pfy, idx_FY, len0_FY, inv_dh_c);

                    #pragma unroll
                    for (std::size_t k = Stage + 1; k < TotalStages; ++k)
                        pphi[k][idx_phi] += kE_here * dt*s_aE[k][Stage];

                    pstate[idx_phi] += kE_here * dt*s_b[Stage];
                }

                #pragma simd
                #pragma ivdep
/*
                for ( auto i = bi
                    ; i <= std::min(bi + CX, phys_upper_phi[0])
                    ; ++i
                    )
*/
                for (auto i = phys_lower_phi[0]; i <= phys_upper_phi[0]; ++i)
                {
//                    auto const idx_kE =
//                        i-lower_kE[0] + len0_kE*(dif1_kE + len1_kE*dif2_kE);
                    auto const idx_phi =
                        i-lower_phi[0] + len0_phi*(dif1_phi + len1_phi*dif2_phi);
                    auto const idx_FZ =
                        i-lower_FZ[0] + len0_FZ*(dif1_FZ + len1_FZ*dif2_FZ);

                    Real kE_here = profile.horizontal_stencil_1d<2>
                        (pfz, idx_FZ, len0_FZ*len1_FZ, inv_dh_c);

                    #pragma unroll
                    for (std::size_t k = Stage + 1; k < TotalStages; ++k)
                        pphi[k][idx_phi] += kE_here * dt*s_aE[k][Stage];

                    pstate[idx_phi] += kE_here * dt*s_b[Stage];
                }
            }
        }

/*
//                ANNOTATE_SITE_BEGIN(explicitOpY);
                #pragma simd
//                #pragma unroll(4)
                #pragma ivdep
                for (auto i = lower[0]; i <= upper[0]; ++i)
                {
                    IntVect h(i, j, k  );

//                    ANNOTATE_ITERATION_TASK(horizontal_stencil_y);
//                    #pragma distribute_point
                    kE(h) = profile.horizontal_stencil_1d<1>(h, FY, inv_dh_c);
                }
//                ANNOTATE_SITE_END();

//                ANNOTATE_SITE_BEGIN(explicitOpZ);
                #pragma simd
//                #pragma unroll(4)
                #pragma ivdep
                for (auto i = lower[0]; i <= upper[0]; ++i)
                {
                    IntVect h(i, j, k  );

//                    ANNOTATE_ITERATION_TASK(horizontal_stencil_z);
//                    #pragma distribute_point
                    kE(h) += profile.horizontal_stencil_1d<2>(h, FZ, inv_dh_c);
                }
//                ANNOTATE_SITE_END();
*/

/*
                auto k = lower[2];

                Real const inv_dh_c = profile.inverse_dh_c();

                #pragma simd
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
                #pragma simd
                for (; k <= upper[2]; ++k)
                {
                    IntVect h0(i, j, k  );

                    kE(h0) = profile.horizontal_stencil(h0, FY, FZ, inv_dh_c);
                }
*/
    } // }}}

// FIXME FIXME FIXME
    template <std::size_t Stage, typename X>
    void implicitOp(
        Real t
      , X&& kI_
      , X&& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
        {
// FIXME FIXME FIXME
//            kI_.U().setVal(0.0);
            return;
        }

        auto& kI  = kI_.U();
        auto& phi = phi_.U();
 
        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        //#pragma omp parallel for schedule(static)
        for (auto j = lower[1]; j <= upper[1]; ++j)
            for (auto k = lower[2]; k <= upper[2]; ++k)
            {
                LapackFactorization A;

                profile.build_vertical_operator_for_apply(A, b);

                profile.apply_vertical_operator(j, k, A, b, kI, phi);
            }
    } // }}}

    template <std::size_t Stage, typename X>
    void solve(
        Real t
      , Real dtscale
      , X&& phi_
        ) const
    { // {{{
        if (implicitOpDisabled())
            return;

        // For HPX, we operate on sub_problem_states (e.g. FArrayBox
        // granularity). For MPI, these exchanges are done on problem_states
        // (e.g. LevelData granularity). 
        //#if defined(CH_HPX)
        //    phi_.exchangeSync();
        //#endif

        auto& phi = phi_.U();

        IntVect lower = phi.smallEnd();
        IntVect upper = phi.bigEnd(); 

        lower.shift(profile.config.ghost_vector);
        upper.shift(-1*profile.config.ghost_vector);

        Box b(lower, upper);

        //#pragma omp parallel for schedule(static)
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

    template <std::size_t Stage, std::size_t TotalStages>
    void explicitOp(
        Real t
      , Real dt
      , Real const s_aE[TotalStages][TotalStages]
      , Real const s_b[TotalStages]
      , problem_state& state_
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

        //#pragma omp parallel for schedule(static)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_state(state_, dit[ibox]);
            sub_problem_state sub_kE(kE_, dit[ibox]);
            sub_problem_state sub_phi(phi_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.explicitOp<Stage>(t, dt, s_aE, s_b
                                     , sub_state, sub_kE, sub_phi);
        }
    } // }}}

    template <std::size_t Stage>
    void implicitOp(
        Real t
      , problem_state& kI_
      , problem_state& phi_
        ) const
    { // {{{
        DataIterator dit = phi_.U.dataIterator();
        std::size_t const nbox = dit.size();

        //#pragma omp parallel for schedule(static)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_phi(phi_, dit[ibox]);
            sub_problem_state sub_kI(kI_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.implicitOp<Stage>(t, sub_kI, sub_phi);
        }
    } // }}}

    template <std::size_t Stage>
    void solve(
        Real t
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

        //#pragma omp parallel for schedule(static)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            sub_problem_state sub_phi(phi_, dit[ibox]);

            imex_per_box_operators<Profile> sub_imex(profile);

            sub_imex.solve<Stage>(t, dtscale, sub_phi);
        }
    } // }}}

  private:
    Profile const profile;
};

}

#endif // CHOMBO_9AA05D23_647F_48CA_ADE1_CB053BABE61E

