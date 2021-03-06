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
  m_isDefined = false;
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
  m_isDefined = true;

  std::size_t num_points = 0;

  DataIterator dit = m_data.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    num_points += m_data[dit].box().numPts();
}

void
TestData::define(const TestData& a_state)
{
  const LevelData<FArrayBox>& srcData = a_state.data();
  m_grids = srcData.disjointBoxLayout();
  m_nComp = srcData.nComp();
  m_ghostVect = srcData.ghostVect();
  m_data.define(m_grids, m_nComp, m_ghostVect);
  m_isDefined = true;
}

/// Constructor that aliases an incoming LevelData<FArrayBox>
void
TestData::aliasData(LevelData<FArrayBox>& a_data)
{
  Interval aliasInt(0,a_data.nComp()-1);
  aliasLevelData<FArrayBox>(m_data, &a_data, aliasInt);
  m_isDefined = true;
}

void
TestData::copy(const std::pair<DataIndex,Box>& a_tileix, 
    const TestData& a_state)
{
  CH_TIMERS("TestData::copy(TestData)");
  DataIndex dataix=a_tileix.first;
  const FArrayBox& srcDataFab = a_state.fab(dataix);
  FArrayBox& dataFab = m_data[dataix];
  // vectorizedCopy(srcDataFab, dataFab, a_tileix.second);
  dataFab.copy(srcDataFab, a_tileix.second);
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
