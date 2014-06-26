#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif


#include <algorithm>

#include "parstream.H"
#include "FArrayBox.H"
#include "LevelData.H"
#include "LevelDataOps.H"
#include "LayoutIterator.H"
#include "FABView.H"
#include "AMRIO.H"

// ARK4 test classes
#include "ARK4.H"
#include "TestOpData.H"
#include "TestImExOp.H"

#include "UsingNamespace.H"

/// Global variables for handling output:
static const char* pgmname = "testARK4" ;
static const char* indent = "   ";
static const char* indent2 = "      " ;
static bool verbose = true ;

/// Prototypes:
int
testARK4();

void
parseTestOptions(int argc ,char* argv[]) ;

int
main(int argc ,char* argv[])
{
#ifdef CH_MPI
  MPI_Init (&argc, &argv);
#endif
  pout () << indent2 << "Beginning " << pgmname << " ..." << std::endl ;

  int status = testARK4();

  if ( status == 0 )
    pout() << indent << pgmname << " passed." << std::endl ;
  else
    pout() << indent << pgmname << " failed with return code " << status << std::endl ;

#ifdef CH_MPI
  MPI_Finalize ();
#endif
  return status ;
}

// Check that the data is a constant, and return the constant
Real ldfabVal(LevelData<FArrayBox>& a_data)
{
  Real ldmin;
  Real ldmax;
  ldmin = CH_BADVAL;
  ldmax = -CH_BADVAL;
  DataIterator dit(a_data.getBoxes());
  for (dit.begin(); dit.ok(); ++dit)
  {
    Real min = a_data[dit].min();
    ldmin = (ldmin < min) ? ldmin : min;
    Real max = a_data[dit].max();
    ldmax = (ldmax > max) ? ldmax : max;
  }
  CH_assert(ldmax == ldmin);
  return ldmax;
}

int
testARK4 ()
{
  // Set up the data classes
  int size = 1;
  Vector<Box> boxes(1,Box(IntVect::Zero, (size-1)*IntVect::Unit));
  Vector<int> procs(1,1);
  DisjointBoxLayout dbl(boxes,procs);
  LevelData<FArrayBox> data(dbl,1,IntVect::Zero);
  LevelData<FArrayBox> accum(dbl,1,IntVect::Zero);
  TestOpData soln;
  // soln.define(dbl,1,IntVect::Zero);
  soln.aliasData(data, &accum);

  int nDenseCoefs = 4;
  Vector<TestOpData*> denseCoefs(nDenseCoefs);
  for (int icoef = 0; icoef < nDenseCoefs; ++icoef)
  {
    denseCoefs[icoef] = new TestOpData();
    denseCoefs[icoef]->define(dbl,1,IntVect::Zero);
  }

  // Do a convergence study, across 3 time step sizes
  Real basedt = 1;
  Real Nres = 4;
  Vector<Real> errors(Nres,0);
  Vector<Real> denseErrs(Nres,0);

  // Exact solution is exp((cE + cI)*t)*(1+t)
  // So that:
  //   explicit op = (1/(1+t) + cE) * phi;
  //   implicit op = cI * phi;
  Real coef = TestImExOp::s_cE + TestImExOp::s_cI;

  bool denseOutput = true;
  ARK4<TestOpData, TestImExOp> ark(soln,basedt, denseOutput); 
  LevelDataOps<FArrayBox> ops;
  for (int res=0; res < Nres; ++res)
  {
    Real time = 0;
    int Nstep = pow(2,res+2);
    Real dt = basedt / (Real) Nstep;
    pout() << "Time step: " << dt << std::endl;
    ark.resetDt(dt);
    // Set the initial condition
    Real phi0 = 1.0;
    ops.setVal(data, phi0);
    ops.setVal(accum, 0);
    // advance nstep
    for (int step = 0; step < Nstep; ++step)
    {
      ark.advance(time, soln);
      time += dt;
    }
    Real exact = exp(coef*time)*(1 + time);
    Real val = ldfabVal(data);
    Real error = (exact - val);
    pout() << "Soln at time " << time << " = " << 
      val << ", error = " << error << std::endl;
    errors[res] = error;

    Real accumDiff = ldfabVal(accum);
    pout() << "Accumulated RHS = " << accumDiff <<
      " , vs. soln change = " << (accumDiff - (val - phi0)) << std::endl;

    // Test dense output for the last time step
    Real theta = 1 - 1/sqrt(2);
    Real tint = time - (1-theta)*dt;
    exact = exp(coef*tint)*(1 + tint);
    ark.denseOutputCoefs(denseCoefs);
    // Compute time-interpolated value in soln
    soln.copy(*denseCoefs[0]);
    for (int icoef = 1; icoef < nDenseCoefs; ++icoef)
    {
      Real factor = pow(theta, icoef);
      soln.increment(*denseCoefs[icoef],factor);
    }
    val = ldfabVal(data);
    error = exact - val;
    denseErrs[res] = error;
    pout() << "Dense output at time " << tint << " = " << 
      val << ", error = " << error << std::endl;
  }

  pout() << "Orders of convergence: " << std::endl;
  Real rate = 4;
  for (int res=1; res < Nres; ++res)
  {
    Real ratio = errors[res] / errors[res-1];
    Real solnrate = (-log(ratio) / log(2));
    pout() << "  soln: " << solnrate << std::endl;
    ratio = denseErrs[res] / denseErrs[res-1];
    Real denserate = (-log(ratio) / log(2));
    pout() << "  dense output: " << denserate << std::endl;
    rate = std::min(solnrate, denserate);
  }

  return (rate > 3.8) ? 0 : 1;
}
