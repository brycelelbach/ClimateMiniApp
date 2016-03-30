#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _TESTDATA_H_
#define _TESTDATA_H_

#include <cassert>

#include "VectorizedFABOps.H"
#include "FArrayBox.H"
#include "LevelData.H"

#include "NamespaceHeader.H"

struct TestData
{
    TestData(
        DisjointBoxLayout const& a_layout
      , int a_nComp
      , IntVect a_ghostVect
      , std::uint64_t a_tile_width
        )
      : m_tiles()
    {
        U.define(a_layout, a_nComp, a_ghostVect);
        A.define(a_layout, a_nComp, a_ghostVect);
        B.define(a_layout, a_nComp, a_ghostVect);
        C.define(a_layout, a_nComp, a_ghostVect);
        makeTiles(a_ghostVect, a_tile_width);
    }

    void exchange()
    {
        U.exchange();
    }

    void plus(TestData const& a_rhs, Real a_factor)
    {
        DataIterator dit = U.dataIterator();
        std::size_t const nbox = dit.size();

        for (std::size_t ibox = 0; ibox < nbox; ++ibox)
        {
            vectorizedPlus(a_rhs.U[dit[ibox]], U[dit[ibox]], a_factor);
        }
    }

    Box domain() const
    {
        return U.disjointBoxLayout().physDomain().domainBox();
    }

    std::vector<std::pair<DataIndex, Box> > const& tiles() const
    {
        return m_tiles;
    }

    LevelData<FArrayBox> U;

    LevelData<FArrayBox> A;
    LevelData<FArrayBox> B;
    LevelData<FArrayBox> C;

  private:
    void makeTiles(IntVect a_ghostVect, std::uint64_t a_tile_width)
    {
        DataIterator dit = U.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            Box b = U.disjointBoxLayout()[dit()];
            b.grow(a_ghostVect);

            assert(0 == (b.size()[1] % a_tile_width));

            for (int i = b.smallEnd(1); i + a_tile_width <= b.bigEnd(1);)
            { 
                Box blo = b;
                // b is "bhi" after chop.
                b = blo.chop(1, i += a_tile_width); 
                m_tiles.push_back(std::pair<DataIndex, Box>(dit(), blo));
            }

            m_tiles.push_back(std::pair<DataIndex, Box>(dit(), b));
        }
    }

    std::vector<std::pair<DataIndex, Box> > m_tiles;
};

#include "NamespaceFooter.H"

#endif

