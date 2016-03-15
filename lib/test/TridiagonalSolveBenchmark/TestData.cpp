#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "TestData.H"

#include "NamespaceHeader.H"

TestData::TestData()
{
}

TestData::~TestData()
{
}

void
TestData::define(/// box layout at this level
                        const DisjointBoxLayout&    a_layout,
                        /// number of conserved components
                        const int                   a_nComp,
                        /// ghost vector
                        const IntVect&              a_ghostVect
                        )
{
  m_grids = a_layout;
  m_ghostVect = a_ghostVect;
  m_nComp = a_nComp;
  m_data.define(m_grids, m_nComp, m_ghostVect);
  m_matrix_a.define(m_grids, m_nComp, IntVect::Zero);
  m_matrix_b.define(m_grids, m_nComp, IntVect::Zero);
  m_matrix_c.define(m_grids, m_nComp, IntVect::Zero);
}

void
TestData::define(const TestData& a_state)
{
  const LevelData<FArrayBox>& srcData = a_state.data();
  m_grids = srcData.disjointBoxLayout();
  m_nComp = srcData.nComp();
  m_ghostVect = srcData.ghostVect();
  m_data.define(m_grids, m_nComp, m_ghostVect);
  m_matrix_a.define(m_grids, m_nComp, IntVect::Zero);
  m_matrix_b.define(m_grids, m_nComp, IntVect::Zero);
  m_matrix_c.define(m_grids, m_nComp, IntVect::Zero);
}

/// Constructor that aliases an incoming LevelData<FArrayBox>
void
TestData::aliasData(LevelData<FArrayBox>& a_data,
                    LevelData<FArrayBox>& a_matrix_a,
                    LevelData<FArrayBox>& a_matrix_b,
                    LevelData<FArrayBox>& a_matrix_c)
{
  {
    Interval aliasInt(0,a_data.nComp()-1);
    aliasLevelData<FArrayBox>(m_data, &a_data, aliasInt);
  }

  {
    Interval aliasInt(0,a_matrix_a.nComp()-1);
    aliasLevelData<FArrayBox>(m_matrix_a, &a_matrix_a, aliasInt);
  }

  {
    Interval aliasInt(0,a_matrix_b.nComp()-1);
    aliasLevelData<FArrayBox>(m_matrix_b, &a_matrix_b, aliasInt);
  }

  {
    Interval aliasInt(0,a_matrix_c.nComp()-1);
    aliasLevelData<FArrayBox>(m_matrix_c, &a_matrix_c, aliasInt);
  }
}

void
TestData::exchange()
{
  CH_TIMERS("TestData::exchange");
  m_data.exchange();
}
  
void
TestData::zero()
{
  CH_TIMERS("TestData::zero");
  DataIterator dit = m_data.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      FArrayBox& dataFab = m_data[dit];
      dataFab.setVal(0);
    }
}

#include "NamespaceFooter.H"
