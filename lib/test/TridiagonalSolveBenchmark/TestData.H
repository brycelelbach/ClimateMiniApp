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

#include "FArrayBox.H"
#include "LevelData.H"
#include "LevelFluxRegister.H"
#include "RealVect.H"

#include "NamespaceHeader.H"

// forward declaration

class TestData
{
public:

  /// Default constructor
  /**
     Object requires define() to be called before all other functions.
  */
  TestData();

  /// Destructor
  virtual ~TestData();

  /// Actual constructor.
  void define(/// box layout at this level
              const DisjointBoxLayout&    a_layout,
              /// number of conserved components
              const int                   a_nComp,
              /// ghost vector
              const IntVect&              a_ghostVect
              );

  void define(const TestData& a_state);

  void aliasData(LevelData<FArrayBox>& a_data,
                 LevelData<FArrayBox>& a_matrix_a,
                 LevelData<FArrayBox>& a_matrix_b,
                 LevelData<FArrayBox>& a_matrix_c
                );

  void exchange();

  void zero();

  const LevelData<FArrayBox>& data() const
  { return m_data; }
  LevelData<FArrayBox>& data()
  { return m_data; }
  const FArrayBox& data(DataIndex a_dataix) const
  { return m_data[a_dataix]; }
  FArrayBox& data(DataIndex a_dataix)
  { return m_data[a_dataix]; }

  const LevelData<FArrayBox>& A() const
  { return m_matrix_a; }
  LevelData<FArrayBox>& A()
  { return m_matrix_a; }
  const FArrayBox& A(DataIndex a_dataix) const
  { return m_matrix_a[a_dataix]; }
  FArrayBox& A(DataIndex a_dataix)
  { return m_matrix_a[a_dataix]; }

  const LevelData<FArrayBox>& B() const
  { return m_matrix_b; }
  LevelData<FArrayBox>& B()
  { return m_matrix_b; }
  const FArrayBox& B(DataIndex a_dataix) const
  { return m_matrix_b[a_dataix]; }
  FArrayBox& B(DataIndex a_dataix)
  { return m_matrix_b[a_dataix]; }

  const LevelData<FArrayBox>& C() const
  { return m_matrix_c; }
  LevelData<FArrayBox>& C()
  { return m_matrix_c; }
  const FArrayBox& C(DataIndex a_dataix) const
  { return m_matrix_c[a_dataix]; }
  FArrayBox& C(DataIndex a_dataix)
  { return m_matrix_c[a_dataix]; }

  Box domain(DataIndex a_dataix) const
  { return m_data.disjointBoxLayout()[a_dataix]; }

  Vector<std::pair<DataIndex,Box> > getTiles(DataIterator& a_dit) const
  {
    Vector<std::pair<DataIndex,Box> > retval;
    for (a_dit.begin(); a_dit.ok(); ++a_dit)
    {
      // FIXME - hardwired 2 boxes per dit. Need to tile here
      Box b = domain(a_dit());
      int chop = b.smallEnd(1) + b.size(1) / 2;
      Box blo = b.chop(1, chop); // b is bhi after chop
      retval.push_back(std::pair<DataIndex,Box>(a_dit(), blo)); 
      retval.push_back(std::pair<DataIndex,Box>(a_dit(), b)); 
    }
    return retval;
  }

protected:

  /// layout for this level
  DisjointBoxLayout m_grids;

  /// number of components
  int m_nComp;

  /// data on this level
  LevelData<FArrayBox> m_data;

  LevelData<FArrayBox> m_matrix_a;
  LevelData<FArrayBox> m_matrix_b;
  LevelData<FArrayBox> m_matrix_c;

  /// ghost vector
  IntVect m_ghostVect;

private:

  // Disallowed for all the usual reasons
  void operator=(const TestData&);
  TestData(const TestData&);
};

#include "NamespaceFooter.H"

#endif
