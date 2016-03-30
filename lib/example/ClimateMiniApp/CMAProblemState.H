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

#include "VectorizedFABOps.H"

#if defined(CH_HPX)
    #include "AsyncLevelData.H"
    #include "AsyncExchange.H"
#else
    #include "LevelData.H"
#endif

namespace climate_mini_app
{

enum {
    PS_TAG_ANONYMOUS = 0,
    PS_TAG_STATE,
    PS_TAG_ANALYTIC, // I think this is unneeded as because never exchange this.
    PS_TAG_PHI0,
    PS_TAG_PHI1,
    PS_TAG_PHI2,
    PS_TAG_PHI3,
    PS_TAG_PHI4,
    PS_TAG_PHI5
};

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

typedef std::pair<DataIndex, Box> tile;

typedef std::vector<tile> tile_vector;

struct tile_range
{
    tile_range() = default;

    tile_range(tile_range const&) = default;
    tile_range(tile_range&&) = default;
    tile_range& operator=(tile_range const&) = default;
    tile_range& operator=(tile_range&&) = default;

    tile_range(tile_vector::const_iterator b, tile_vector::const_iterator e)
    // {{{
      : begin_(b)
      , end_(e)
    {} // }}}

    tile_vector::const_iterator begin() const
    { // {{{
        return begin_;
    } // }}}

    tile_vector::const_iterator end() const
    { // {{{
        return end_;
    } // }}}

  private:
    tile_vector::const_iterator begin_;
    tile_vector::const_iterator end_;
};

typedef std::map<DataIndex, tile_range> tile_map;

struct problem_state_fab;

struct problem_state_tile;

struct problem_state
{
    problem_state()
    // {{{
      : U()
      , FX()
      , FY()
      , tile_vector_()
      , tile_map_()
      #if defined(CH_HPX)
      , epoch_()
      #endif
    {} // }}}

    problem_state(
        DisjointBoxLayout const& dbl
      , int comps
      , IntVect ghost
      , std::uint64_t tile_width
      , std::size_t tag = PS_TAG_ANONYMOUS
        )
    // {{{
      : U()
      , FX()
      , FY()
      , tile_vector_()
      , tile_map_()
      #if defined(CH_HPX)
      , epoch_()
      #endif
    { 
        define(dbl, comps, ghost, tile_width, tag);
    } // }}}

    void define(
        DisjointBoxLayout const& dbl
      , int comps
      , IntVect ghost
      , std::uint64_t tile_width
      , std::size_t tag = PS_TAG_ANONYMOUS
        )
    { // {{{
        #if defined(CH_HPX)
            U.define(dbl, comps, ghost, tag);
    
            auto defineFX =
                [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
                {
                    Box b = ld.disjointBoxLayout()[di];
                    b.grow(ld.ghostVect());
                    b = surroundingNodes(b, 0);
                    ld[di].define(b, ld.nComp());
                }; 

            auto defineFY =
                [] (AsyncLevelData<FArrayBox>& ld, DataIndex di)
                {
                    Box b = ld.disjointBoxLayout()[di];
                    b.grow(ld.ghostVect());
                    b = surroundingNodes(b, 1);
                    ld[di].define(b, ld.nComp());
                }; 

            FX.define(dbl, comps, ghost, PS_TAG_ANONYMOUS, defineFX);
            FY.define(dbl, comps, ghost, PS_TAG_ANONYMOUS, defineFY);
            
            epoch_.define(dbl);
    
            DataIterator dit(U.dataIterator());
    
            for (dit.begin(); dit.ok(); ++dit)
                epoch_[dit()] = 0;
        #else
            U.define(dbl, comps, ghost);

            FluxFABDataFactory fluxFactoryX(0), fluxFactoryY(1);
    
            FX.define(dbl, comps, ghost, fluxFactoryX);
            FY.define(dbl, comps, ghost, fluxFactoryY);
        #endif

        make_tiles(ghost, tile_width);
    } // }}}

    DataIterator dataIterator()
    { // {{{
        return U.dataIterator();
    } // }}}

    void copy(problem_state const& X)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t const nbox = dit.size();

        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
            U[dit[ibox]].copy(X.U[dit[ibox]]);
    } // }}}

    void setVal(Real val)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t const nbox = dit.size();

        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            U[dit[ibox]].setVal(val);
        }
    } // }}}

    void plus(problem_state const& X, Real factor)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t const nbox = dit.size();

        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
            U[dit[ibox]].plus(X.U[dit[ibox]], factor);
    } // }}}

    void plus(problem_state const& X)
    { // {{{
        DataIterator dit = U.dataIterator();
        std::size_t const nbox = dit.size();

        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
            U[dit[ibox]].plus(X.U[dit[ibox]]);
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

    tile_range tiles() const
    { // {{{
        return tile_range(tile_vector_.begin(), tile_vector_.end());
    } // }}}

    tile_range tiles(std::size_t ibox) const
    { // {{{
        DataIterator dit(U.dataIterator());

        std::size_t const nbox = dit.size();

        assert(ibox <= nbox);

        return tile_map_.at(dit[ibox]);
    } // }}}

    tile_range tiles(DataIndex di) const
    { // {{{
        return tile_map_.at(di);
    } // }}}

    problem_state_fab operator[](std::size_t ibox);

    problem_state_tile operator[](tile t);

    friend struct problem_state_fab;

    friend struct problem_state_tile;

  public:
    // FIXME FIXME FIXME: Make these private and create accessors for
    // interface compatiblity with problem_state_*
#if defined(CH_HPX)
    AsyncLevelData<FArrayBox> U;
    AsyncLevelData<FArrayBox> FX;
    AsyncLevelData<FArrayBox> FY;

    AsyncLevelData<FArrayBox> A;
    AsyncLevelData<FArrayBox> B;
    AsyncLevelData<FArrayBox> C;
#else
    LevelData<FArrayBox> U;
    LevelData<FArrayBox> FX;
    LevelData<FArrayBox> FY;

    LevelData<FArrayBox> A;
    LevelData<FArrayBox> B;
    LevelData<FArrayBox> C;
#endif

  private:
    void make_tiles(IntVect ghost, std::uint64_t tile_width) 
    { // {{{
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            Box b = U.disjointBoxLayout()[dit()];
            b.grow(ghost);

            assert(0 == (b.size()[1] % tile_width));

            std::size_t const begin_idx = tile_vector_.size();

            std::size_t end_idx = begin_idx;

            for (int i = b.smallEnd(1); i + tile_width <= b.bigEnd(1); ++end_idx)
            { 
                Box blo = b;
                // b is "bhi" after chop.
                b = blo.chop(1, i += tile_width); 
                tile_vector_.emplace_back(dit(), blo);
            }

            tile_vector_.emplace_back(dit(), b);
            ++end_idx;

            auto begin_it = tile_vector_.begin();
            std::advance(begin_it, begin_idx);

            auto end_it = tile_vector_.begin();
            std::advance(end_it, end_idx);

            std::pair<DataIndex, tile_range> p(
                dit(), tile_range(begin_it, end_it)
            );

            bool r = tile_map_.insert(p).second;

            assert(r);
        }
    } // }}}

    tile_vector tile_vector_;
    tile_map tile_map_;

#if defined(CH_HPX)
    LayoutData<std::size_t> epoch_;
#endif
};

struct problem_state_fab
{
    problem_state_fab()
    // {{{
      : ps_(0)
      , di_()
      , tr_()
    {} // }}}

    problem_state_fab(problem_state& ps, DataIndex di)
    // {{{
      : ps_(&ps)
      , di_(di)
      , tr_(ps.tiles(di))
    {
        assert(tr_.begin() != tr_.end());
    } // }}}

    problem_state_fab(problem_state_fab const& other) = default; 

    problem_state_fab& operator=(problem_state_fab const& other) = default;

    void copy(problem_state_fab const& X)
    { // {{{
        vectorizedCopy(X.U(), U());
    } // }}}

    // FIXME: Vectorize.
    void setVal(Real val)
    { // {{{
        U().setVal(val);
    } // }}}

    void plus(problem_state_fab const& X, Real factor)
    { // {{{
        vectorizedPlus(X.U(), U(), factor);
    } // }}}

    void plus(problem_state_fab const& X)
    { // {{{
        vectorizedPlus(X.U(), U());
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

    Box box() const
    { // {{{
        assert(ps_);
        return ps_->U[di_].box(); 
    } // }}}

    tile_range tiles() const
    { // {{{
        return tr_; 
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

    FArrayBox& FX()
    { // {{{
        assert(ps_);
        return ps_->FX[di_]; 
    } // }}}

    FArrayBox const& FX() const
    { // {{{
        assert(ps_);
        return ps_->FX[di_]; 
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

  private:
    problem_state* ps_;
    DataIndex di_;
    tile_range tr_;
};

struct problem_state_tile
{
    problem_state_tile()
    // {{{
      : ps_(0)
      , tile_()
    {} // }}}

    problem_state_tile(problem_state& ps, tile t)
    // {{{
      : ps_(&ps)
      , tile_(t)
    {
    } // }}}

    problem_state_tile(problem_state_tile const& other) = default; 

    problem_state_tile& operator=(problem_state_tile const& other) = default;

    // FIXME: Vectorize.
    void copy(problem_state_tile const& X)
    { // {{{
        assert(tile_.second == X.tile_.second);
        U().copy(X.U(), tile_.second, 0, tile_.second, 0, 1);
    } // }}}

    // FIXME: Vectorize.
    void setVal(Real val)
    { // {{{
        U().setVal(val, tile_.second, 0);
    } // }}}

    // FIXME: Vectorize.
    void plus(problem_state_tile const& X, Real factor)
    { // {{{
        assert(tile_.second == X.tile_.second);
        U().plus(X.U(), tile_.second, tile_.second, factor, 0, 0, 1);
    } // }}}

    // FIXME: Vectorize.
    void plus(problem_state_tile const& X)
    { // {{{
        assert(tile_.second == X.tile_.second);
        U().plus(X.U(), tile_.second, tile_.second, 0, 0, 1);
    } // }}}

#if defined(CH_HPX)
    hpx::future<void> exchangeAsync()
    { // {{{
        assert(ps_);
        return LocalExchangeAsync(ps_->epoch_[tile_.first]++, tile_.first, ps_->U);
    } // }}}

    void exchangeSync()
    { // {{{
        assert(ps_);
        LocalExchangeSync(ps_->epoch_[tile_.first]++, tile_.first, ps_->U);
    } // }}}
#endif

    Box box() const
    { // {{{
        return tile_.second; 
    } // }}}

    FArrayBox& U()
    { // {{{
        assert(ps_);
        return ps_->U[tile_.first]; 
    } // }}}

    FArrayBox const& U() const
    { // {{{
        assert(ps_);
        return ps_->U[tile_.first]; 
    } // }}}

    FArrayBox& FX()
    { // {{{
        assert(ps_);
        return ps_->FX[tile_.first]; 
    } // }}}

    FArrayBox const& FX() const
    { // {{{
        assert(ps_);
        return ps_->FX[tile_.first]; 
    } // }}}

    FArrayBox& FY()
    { // {{{
        assert(ps_);
        return ps_->FY[tile_.first]; 
    } // }}}

    FArrayBox const& FY() const
    { // {{{
        assert(ps_);
        return ps_->FY[tile_.first]; 
    } // }}}

  private:
    problem_state* ps_;
    tile tile_;
};

problem_state_fab problem_state::operator[](std::size_t ibox)
{ // {{{
    DataIterator dit(U.dataIterator());

    std::size_t const nbox = dit.size();

    assert(ibox <= nbox);

    return problem_state_fab(*this, dit[ibox]);
} // }}}

problem_state_tile problem_state::operator[](tile t)
{ // {{{
    return problem_state_tile(*this, t);
} // }}}

}

#endif // CHOMBO_B7CAA48C_980E_4465_A464_5762F6DD5B09

