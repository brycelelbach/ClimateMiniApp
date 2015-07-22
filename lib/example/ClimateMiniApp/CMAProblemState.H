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

#if !defined(CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09)
#define CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09

#if defined(CH_HPX)
    #include <hpx/lcos/when_all.hpp>
#endif

#include <cstdint>

#include <assert.h>

#include "REAL.H"
#include "IntVect.H"
#include "FArrayBox.H"

#if defined(CH_HPX)
    #include "AsyncLevelData.H"
    #include "AsyncExchange.H"
#else
    #include "LevelData.H"
#endif

#include "LapackFactorization.H"

namespace climate_mini_app
{

#if !defined(CH_HPX)
/// Creates FArrayBoxes suitable for storing fluxes for the given direction.
struct FluxFABDataFactory : DataFactory<FArrayBox>
{
    FluxFABDataFactory(std::size_t dir)
    // {{{
      : dir_(dir)
    {} // }}} 

    FArrayBox* create(Box const& b, int ncomps, DataIndex const& a_datInd) const
    { // {{{
        Box tmp = surroundingNodes(b, dir_);
        return new FArrayBox(tmp, ncomps);
    } // }}}

  private:
    std::size_t dir_;
};
#endif

struct problem_state
{
    problem_state()
    // {{{
      : A0()
      , A1()
      , U()
      , FY()
      , FZ()
      #if defined(CH_HPX)
      , epoch_()
      #endif
    {} // }}}

    problem_state(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    // {{{
      : A0()
      , A1()
      , U()
      , FY()
      , FZ()
      #if defined(CH_HPX)
      , epoch_()
      #endif
    { 
        define(dbl, comps, ghost);
    } // }}}

    void define(DisjointBoxLayout const& dbl, int comps, IntVect ghost)
    { // {{{
        A0.define(dbl);
        A1.define(dbl);

        U.define(dbl, comps, ghost);

        #if defined(CH_HPX)
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
        #else
            FluxFABDataFactory factY(1), factZ(2);
    
            FY.define(dbl, comps, ghost, factY);
            FZ.define(dbl, comps, ghost, factZ);
        #endif
    } // }}}

    void copy(problem_state const& X)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            U[dit[ibox]].copy(X.U[dit[ibox]]);
            FY[dit[ibox]].copy(X.FY[dit[ibox]]);
            FZ[dit[ibox]].copy(X.FZ[dit[ibox]]);
        }
    } // }}}

    void setVal(Real val)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            U[dit[ibox]].setVal(val);
            FY[dit[ibox]].setVal(val);
            FZ[dit[ibox]].setVal(val);
        }
    } // }}}

    void plus(problem_state const& X, Real factor)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            U[dit[ibox]].plus(X.U[dit[ibox]], factor);
        }
    } // }}}

    void plus(problem_state const& X)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t nbox = dit.size();

        #pragma omp parallel for schedule(dynamic)
        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            U[dit[ibox]].plus(X.U[dit[ibox]]);
        }
    } // }}}

#if defined(CH_HPX)
    hpx::future<void> exchangeAsync()
    { // {{{
        DataIterator dit = U.dataIterator();

        std::vector<hpx::future<void> > exchanges;
    
        for (dit.begin(); dit.ok(); ++dit)
            exchanges.push_back(LocalExchangeAsync(epoch_[dit()]++, dit(), U));

        return hpx::lcos::when_all(exchanges);
    } // }}}
#endif

    void exchangeSync()
    { // {{{
        #if defined(CH_HPX)
            exchangeAsync().get();
        #else
            U.exchange();
        #endif
    } // }}}

    friend struct sub_problem_state;

    LayoutData<LapackFactorization> A0;
    LayoutData<LapackFactorization> A1;

#if defined(CH_HPX)
    AsyncLevelData<FArrayBox> U;
    AsyncLevelData<FArrayBox> FY;
    AsyncLevelData<FArrayBox> FZ;

  private:
    LayoutData<std::size_t> epoch_;
#else
    LevelData<FArrayBox> U;
    LevelData<FArrayBox> FY;
    LevelData<FArrayBox> FZ;
#endif
};

struct sub_problem_state
{
    sub_problem_state()
    // {{{
      : ps_(0)
      , di_()
    {} // }}}

    sub_problem_state(problem_state& ps, DataIndex di)
    // {{{
      : ps_(&ps)
      , di_(di)
    {} // }}}

    sub_problem_state(sub_problem_state const& other)
    // {{{
      : ps_(other.ps_)
      , di_(other.di_)
    {} // }}}

    sub_problem_state& operator=(sub_problem_state const& other)
    { // {{{
        ps_ = other.ps_;
        di_ = other.di_;
        return *this;
    } // }}}

    void copy(sub_problem_state const& X)
    { // {{{
        U().copy(X.U());
        FY().copy(X.FY());
        FZ().copy(X.FZ());
    } // }}}

    void setVal(Real val)
    { // {{{
        U().setVal(val);
        FY().setVal(val);
        FZ().setVal(val);
    } // }}}

    void plus(sub_problem_state const& X, Real factor)
    { // {{{
        U().plus(X.U(), factor);
    } // }}}

    void plus(sub_problem_state const& X)
    { // {{{
        U().plus(X.U());
    } // }}}

#if defined(CH_HPX)
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
#endif

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

    LapackFactorization& A0()
    { // {{{
        assert(ps_);
        return ps_->A0[di_]; 
    } // }}}

    LapackFactorization const& A0() const
    { // {{{
        assert(ps_);
        return ps_->A0[di_]; 
    } // }}}

    LapackFactorization& A1()
    { // {{{
        assert(ps_);
        return ps_->A1[di_]; 
    } // }}}

    LapackFactorization const& A1() const
    { // {{{
        assert(ps_);
        return ps_->A1[di_]; 
    } // }}}

  private:
    problem_state* ps_;
    DataIndex di_;
};

}

#endif // CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09

