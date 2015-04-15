#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <hpx/async.hpp>
#include <hpx/lcos/when_all.hpp>

#include "DisjointBoxLayout.H"
#include "LevelData.H"
#include "BaseFab.H"
#include "REAL.H"
#include "DataIterator.H"
#include "Tuple.H"
#include "InterpF_F.H"
#include "AverageF_F.H"

#include "AsyncFineInterp.H"
#include "AsyncExchange.H"
#include "Ranges.H"

#include "NamespaceHeader.H"

void AsyncFineInterp::define(
    DisjointBoxLayout const& coarseDbl
  , DisjointBoxLayout const& fineDbl
  , int numComps
  , int refRatio
  , IntVect ghost
    )
{
    m_ref_ratio = refRatio;

    // Create the work array.
    DisjointBoxLayout cf_dbl;
    coarsen(cf_dbl, fineDbl, m_ref_ratio);

    m_cf_data.define(cf_dbl, numComps, IntVect::Unit);

    // Create the copier.
    m_interpCopier.unilateralDefine(
        coarseDbl                          // from
      , cf_dbl                             // to
      , coarseDbl.physDomain()             // from domain 
      , ghost 
      , IntVect::Zero
      , coarseDbl.getLocalProcID()         // from procID
    ); 

    // FIXME: Install regions
}

void AsyncFineInterp::pwcInterpToFine(
    std::size_t epoch
  , DataIndex di 
  , AsyncLevelData<FArrayBox> const& coarseData // from
  , AsyncLevelData<FArrayBox>& fineData         // to
  , bool averageFromDest
    )
{
    if (averageFromDest)
    {
        FArrayBox const& fineFab = fineData[di];
        FArrayBox&       crseFab = m_cf_data[di];
        Box const&       crseBox = m_cf_data.disjointBoxLayout()[di];

        Box refbox(IntVect::Zero, (m_ref_ratio-1)*IntVect::Unit);
        FORT_AVERAGE(CHF_FRA(crseFab),
                     CHF_CONST_FRA(fineFab),
                     CHF_BOX(crseBox),
                     CHF_CONST_INT(m_ref_ratio),
                     CHF_BOX(refbox));
    }

    LocalCopySync(
        Comm_CoarseToFineState
      , epoch
      , di
      , coarseData // from
      , m_cf_data  // to
      , m_interpCopier
    );

    FArrayBox const& cfFab   = m_cf_data[di];
    FArrayBox&       fineFab = fineData[di];
    Box const&       cfBox   = m_cf_data.disjointBoxLayout()[di];

    pwcInterpGridData(cfFab, fineFab, cfBox);
}

void AsyncFineInterp::pwcInterpGridData(
    FArrayBox const& coarseFab // from
  , FArrayBox& fineFab         // to
  , Box const& cfBox
    ) const
{
  // Fill fine data with piecewise constant coarse data.
  Box refbox(IntVect::Zero, (m_ref_ratio-1)*IntVect::Unit);

  FORT_INTERPCONSTANT(CHF_FRA(fineFab),
                      CHF_CONST_FRA(coarseFab),
                      CHF_BOX(cfBox),
                      CHF_CONST_INT(m_ref_ratio),
                      CHF_BOX(refbox));
}

#include "NamespaceFooter.H"

