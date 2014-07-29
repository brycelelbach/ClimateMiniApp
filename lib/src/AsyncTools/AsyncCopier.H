#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef CH_25C6D1DE_70C5_4627_940D_5C8060914A53 
#define CH_25C6D1DE_70C5_4627_940D_5C8060914A53 

#include "DisjointBoxLayout.H"
#include "Vector.H"
#include "ProblemDomain.H"

#include "NamespaceHeader.H"

struct AsyncSendInstructions
{
  DataIndex fromIndex;
  DataIndex toIndex;
  Box fromRegion;
  Box toRegion;

  AsyncSendInstructions(const DataIndex& a_from
                      , const DataIndex& a_to
                      , const Box&       a_fromRegion
                      , const Box&       a_toRegion)
    : fromIndex(a_from), toIndex(a_to)
    , fromRegion(a_fromRegion), toRegion(a_toRegion)
  {}

  bool operator<(AsyncSendInstructions const& rhs) const
  {
    return fromIndex < rhs.fromIndex;
  }

  bool operator==(AsyncSendInstructions const& rhs) const
  {
    return fromIndex == rhs.fromIndex;
  }
};

struct AsyncRegion
{
  DataIndex toIndex;
  Box toRegion;

  AsyncRegion(const DataIndex& a_to
            , const Box&       a_toRegion)
    : toIndex(a_to)
    , toRegion(a_toRegion)
  {}

  bool operator<(AsyncRegion const& rhs) const
  {
    return toIndex < rhs.toIndex;
  }

  bool operator==(AsyncRegion const& rhs) const
  {
    return toIndex == rhs.toIndex;
  }
};

class AsyncCopier
{
public:
  AsyncCopier()
    : m_senderMotionPlan()
    , m_regions()
    , m_src()
    , m_dest()
    , m_procID() 
  {}

  ~AsyncCopier()
  {
    clear();
  }

  void clear()
  {
    m_senderMotionPlan.clear();
    m_regions.clear();

    m_src  = BoxLayout();
    m_dest = BoxLayout();

    m_procID = -1;  
  }

  void unilateralDefine(const BoxLayout& a_src,
                        const BoxLayout& a_dest,
                        const ProblemDomain& a_domain,
                        const IntVect& a_destGhost,
                        IntVect a_shift = IntVect::Zero,
                        int a_procID = -1);

  void exchangeDefine(const DisjointBoxLayout& a_grids,
                      const IntVect& a_ghost,
                      int a_procID = -1);

  // Using Boost.Intrusive would be more efficienct storage wise (and probably
  // faster), but I want to avoid adding a direct dependency on Boost for
  // something this generic.

  typedef std::multiset<
    AsyncSendInstructions // from DataIndex, to DataIndex, from (Box), to (Box) 
  > motionPlanSet; 

  typedef std::multiset<
    AsyncRegion           // to DataIndex, to (Box)
  > regionsSet;

  motionPlanSet& senderMotionPlan()
  {
    return m_senderMotionPlan; 
  }

  regionsSet& regions()
  {
    return m_regions;
  }

protected:
  // NOTE: Items that were in fromMotionPlan or localMotionPlan must be in here.
  motionPlanSet m_senderMotionPlan; 

  // NOTE: Items that were in toMotionPlan or localMotionPlan must be in here.
  regionsSet m_regions;

  BoxLayout m_src;  // aka "from"
  BoxLayout m_dest; // aka "to"

  int m_procID;
};

#include "NamespaceFooter.H"
#endif
