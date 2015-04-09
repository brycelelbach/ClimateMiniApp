#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef CH_2188EC9D_E253_4FAD_93EB_DC517EB6464A
#define CH_2188EC9D_E253_4FAD_93EB_DC517EB6464A

#include "REAL.H"
#include "LevelData.H"
#include "BaseFab.H"
#include "FArrayBox.H"
#include "ProblemDomain.H"

#include "AsyncLevelData.H"

#include "NamespaceHeader.H"

class DisjointBoxLayout;

struct AsyncFineInterp
{
public:
  AsyncFineInterp() {}

  ~AsyncFineInterp() {} 

  AsyncFineInterp(
    DisjointBoxLayout const& coarseDbl
  , DisjointBoxLayout const& fineDbl
  , int numComps
  , int refRatio
  , IntVect ghost = IntVect::Zero
    )
  {
    define(coarseDbl, fineDbl, numComps, refRatio, ghost);
  } 

  void define(
    DisjointBoxLayout const& coarseDbl
  , DisjointBoxLayout const& fineDbl
  , int numComps
  , int refRatio
  , IntVect ghost = IntVect::Zero
    );

  /// Piecewise-constant interpolation.
  void pwcInterpToFine(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<FArrayBox> const& coarseData // from
  , AsyncLevelData<FArrayBox>& fineData         // to
  , bool averageFromDest = false
    );

private:
  void pwcInterpGridData(
    FArrayBox const& coarseFab // from
  , FArrayBox& fineFab         // to
  , Box const& cfBox
    ) const;

  int m_ref_ratio;
  AsyncLevelData<FArrayBox> m_cf_data;
  AsyncCopier m_interpCopier;
};

#include "NamespaceFooter.H"
#endif

