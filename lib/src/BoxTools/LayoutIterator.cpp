#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <map>

#include "LayoutIterator.H"
#include "DataIterator.H"
#include "NamespaceHeader.H"

void LayoutIterator::reset()
{
  m_current  = 0;
}

void LayoutIterator::end()
{
  m_current = m_layout.size();
}

LayoutIterator::LayoutIterator(const BoxLayout& a_boxlayout,
                               const int*       a_layoutID)
  :
  m_layout(a_boxlayout),
  m_indices(new Vector<LayoutIndex>()),
  m_current(0)
{
  Vector<LayoutIndex>& indices = *m_indices;
  indices.resize(a_boxlayout.size());
  std::map<int, int> datInd;
  for (int ibox = 0; ibox < a_boxlayout.size(); ibox++)
    {
      LayoutIndex& current = indices[ibox];
      current.m_layoutIntPtr = a_layoutID;
      current.m_index        = ibox;
      current.m_datInd       = datInd[m_layout.procID(current)]++;
    }
}
#include "NamespaceFooter.H"
