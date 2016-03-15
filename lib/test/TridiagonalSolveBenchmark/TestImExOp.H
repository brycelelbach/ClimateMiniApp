#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _TESTIMEXOP_H_
#define _TESTIMEXOP_H_

#include "FArrayBox.H"
#include "LevelData.H"
#include "TestData.H"

#include "NamespaceHeader.H"

struct TestImExOp
{
    static constexpr Real cI = 0.3;
    static constexpr Real kx = 2.0 * M_PI;
    static constexpr Real ky = 2.0 * M_PI;
    static constexpr Real kz = 1.0 * M_PI;

    TestImExOp(Real a_dt)
      : m_dt(a_dt)
    {}

    void exact(TestData& a_exact, Real a_time) const;

    Real advance(TestData& a_state, Real a_time) const
    {
        // Implicit backward Euler solve
        Real t = a_time + m_dt;

        a_state.exchange();

        std::vector<std::pair<DataIndex, Box> > const& tiles = a_state.tiles();

        #pragma omp parallel for schedule(static) 
        for (int tix = 0; tix < tiles.size(); ++tix) 
        {
            implicitSolve(tiles[tix], a_state, t);
        }

        return t;
    }

    void implicitSolve(
        std::pair<DataIndex, Box> const& a_tile
      , TestData& a_state
      , Real a_time
        ) const;

    void buildTridiagonal(
        FArrayBox& a_A
      , FArrayBox& a_B
      , FArrayBox& a_C
      , Box a_box
      , Real a_dx
      , Real a_c1
      , Real a_c2
        ) const;

    void implicitSolveTridiag(
        FArrayBox& a_A
      , FArrayBox& a_B
      , FArrayBox& a_C
      , FArrayBox& a_U
      , Box a_box
        ) const;

  private:
    Real m_dt;
};

#include "NamespaceFooter.H"

#endif
