/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */

////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach aka wash <blelbach@cct.lsu.edu>
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(CHOMBO_B541E23D_4C47_4C2F_8FDC_2FAEE6E2CE81)
#define CHOMBO_B541E23D_4C47_4C2F_8FDC_2FAEE6E2CE81

#include "LayoutData.H"

#include "AsyncCopier.H"
#include "DataflowBox.H"

#include "NamespaceHeader.H"

template <typename T>
struct AsyncLevelData : LayoutData<DataflowBox<T> >
{
    typedef LayoutData<DataflowBox<T> > base_type;

  private:
    DisjointBoxLayout m_disjointBoxLayout;
    IntVect           m_ghost;
    AsyncCopier       m_exchangeCopier;

  public:
    AsyncLevelData()
      : m_disjointBoxLayout()
      , m_ghost()
      , m_exchangeCopier()
    {}

    AsyncLevelData(
        DisjointBoxLayout const& dbl
      , IntVect ghost
        )
    {
        define(dbl, ghost);
    }

    void define(
        DisjointBoxLayout const& dbl
      , IntVect ghost
        )
    {
        CH_assert(dbl.isClosed());

        m_disjointBoxLayout = dbl;
        m_ghost = ghost;

        this->base_type::define(dbl);

        m_exchangeCopier.exchangeDefine(dbl, ghost, dbl.getLocalProcID()); 

        installRegions();
    }

  private:
    void installRegions()
    {
/*
        // Interior region.
        DataIterator dit = this->dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            Box b = m_disjointBoxLayout[dit()];

            (*this)[dit()].addRegion(b);
        }
*/

        // Same-level ghost zone regions.
        for (AsyncRegion const& ar : m_exchangeCopier.regions())
            (*this)[ar.toIndex].addRegion(Comm_ExchangeState, ar.toRegion);
    } 

  public:
    DisjointBoxLayout const& disjointBoxLayout() const
    {
        return m_disjointBoxLayout;
    }

    AsyncCopier const& exchangeCopier() const
    {
        return m_exchangeCopier;
    }

    IntVect ghostVect() const
    {
        return m_ghost;
    }
};

// TODO: Move this into constructor?
template <typename T>
void DefineData(
    AsyncLevelData<T>& ld
  , int comps = 1
    )
{
    DataIterator dit(ld.dataIterator());
    for (dit.begin(); dit.ok(); ++dit)
    {
        Box b = ld.disjointBoxLayout()[dit()];
        b.grow(ld.ghostVect());
        ld[dit()].define(b, comps);
    } 
}

template <typename T>
void DefineData(
    LayoutData<T>& ld
  , int comps = 1
  , IntVect ghost = IntVect::Zero
    )
{
    DataIterator dit(ld.dataIterator());
    for (dit.begin(); dit.ok(); ++dit)
    {
        Box b = ld.boxLayout()[dit()];
        b.grow(ghost);
        ld[dit()].define(b, comps);
    } 
}

#include "NamespaceFooter.H"

#endif // CHOMBO_B541E23D_4C47_4C2F_8FDC_2FAEE6E2CE81
