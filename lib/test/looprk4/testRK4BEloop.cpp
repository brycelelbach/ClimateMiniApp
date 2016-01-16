#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif


#include "parstream.H"
using std::endl;
#include "FArrayBox.H"
#include "LevelData.H"
#include "LevelDataOps.H"
#include "LayoutIterator.H"
#include "LoadBalance.H"
#include "BRMeshRefine.H"
#include "CH_Timer.H"

// ImExRK4BE test classes
#include "ImExRK4BE.H"

#include "TestRhsData.H"
#include "TestSolnData.H"
#include "TestImExOp.H"
#include "FABView.H"
#include "FABView.H"

#include "UsingNamespace.H"

/// Global variables for handling output:
static const char* pgmname = "testRK4BEloop" ;
static const char* indent = "   ";
static const char* indent2 = "      " ;

/// Prototypes:
int
testImExRK4BE();

void
parseTestOptions(int argc ,char* argv[]) ;

int
main(int argc ,char* argv[])
{
#ifdef CH_MPI
  MPI_Init (&argc, &argv);
#endif
  pout () << indent2 << "Beginning " << pgmname << " ..." << endl ;

  int status = testImExRK4BE();

  if ( status == 0 )
    pout() << indent << pgmname << " passed." << endl ;
  else
    pout() << indent << pgmname << " failed with return code " << status << endl ;

#ifdef CH_MPI
  CH_TIMER_REPORT();
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
  DisjointBoxLayout dbl = a_data.getBoxes();
  DataIterator dit(dbl);
  for (dit.begin(); dit.ok(); ++dit)
  {
    Box b = dbl[dit];
    Real min = a_data[dit].min(b);
    ldmin = (ldmin < min) ? ldmin : min;
    Real max = a_data[dit].max(b);
    ldmax = (ldmax > max) ? ldmax : max;
  }
  CH_assert(ldmax == ldmin);
  return ldmax;
}

int
testImExRK4BE ()
{
  CH_TIMERS("testImExRK4BE");
  CH_TIMER("setup",t1);
  CH_TIMER("integrate",t2);
  CH_TIMER("calc error",t3);
  CH_START(t1);

  IntVect numCells(D_DECL(128,128,32));
  IntVect loVect = IntVect::Zero;
  IntVect hiVect = numCells-IntVect::Unit;
  Box domainBox(loVect, hiVect);
  ProblemDomain baseDomain(domainBox);

  int maxBoxSize = 32;
  Vector<Box> vectBoxes;
  domainSplit(baseDomain, vectBoxes, maxBoxSize, 1);
  Vector<int> procAssign(vectBoxes.size(), 0);
  LoadBalance(procAssign, vectBoxes);
  DisjointBoxLayout dbl(vectBoxes, procAssign, baseDomain);

  // Set up the data classes
  IntVect nGhosts(D_DECL(4,4,0));
  LevelData<FArrayBox> data(dbl,1,nGhosts);
  LevelData<FArrayBox> accum(dbl,1,IntVect::Zero);
  TestSolnData soln;
  // soln.define(dbl,1,IntVect::Zero);
  soln.aliasData(data, &accum);

  // int nDenseCoefs = 4;
  // Vector<RefCountedPtr<TestRhsData> > denseCoefs(nDenseCoefs);
  // for (int icoef = 0; icoef < nDenseCoefs; ++icoef)
  // {
  //   denseCoefs[icoef] = new TestRhsData();
  //   denseCoefs[icoef]->define(dbl,1,IntVect::Zero);
  // }

  // Do a convergence study, across 3 time step sizes
  Real basedt = sqrt(.5);
  // Real Nres = 5;
  Real Nres = 1;
  Vector<Real> errors(Nres,0);
  Vector<Real> denseErrs(Nres,0);

  // Exact solution is exp((cE + cI + cS)*t)
  // So that:
  //   explicit op = cE * phi;
  //   implicit op = cI * phi;
  //   split explicit op = cS * phi;

  bool denseOutput = false;
  ImExRK4BE<TestSolnData, TestRhsData, TestImExOp> imex;
  imex.define(soln, basedt, denseOutput); 
  RefCountedPtr<TestImExOp> imexOp = imex.getImExOp();
  LevelDataOps<FArrayBox> ops;
  bool passes = true;
  Real err_tol = 1e-15;
  CH_STOP(t1);
  for (int res=0; res < Nres; ++res)
  {
    CH_START(t2);
    Real time = 0;
    // int Nstep = pow((Real) 2,(Real) res+2);
    int Nstep = 10;
    Real dt = basedt / (Real) Nstep;
    pout() << "Time step: " << dt << endl;
    imex.resetDt(dt);
    // Set the initial condition
    Real phi0 = 1.0;
    ops.setVal(data, phi0);
    // ops.setVal(accum, 0);
    // advance nstep
    for (int step = 0; step < Nstep; ++step)
    {
      imex.advance(time, soln);
      time += dt;
    }
    CH_STOP(t2);
    CH_START(t3);
    Real exact = imexOp->exact(time);
    Real val = ldfabVal(data);
    Real error = (exact - val);
    pout() << "Soln at time " << time << " = " << 
      val << ", error = " << error << endl;
    errors[res] = error;
    passes = (error <= err_tol) && passes;
    CH_STOP(t3);

    // Real accumDiff = ldfabVal(accum);
    // pout() << "Accumulated RHS = " << accumDiff <<
    // " , difference from soln change = " << (accumDiff - (val - phi0)) << endl;

#if 0
    // Test dense output for the last time step
    Real theta = 1 - 1/sqrt(2);
    // Real theta = 1;
    Real tint = time - (1-theta)*dt;
    exact = imexOp->exact(tint);
    pout() << "  exact value " << exact << endl;
    // Vector<TestRhsData*> coefs = imex->denseOutputCoefs();

    // Compute time-interpolated value in soln
    // soln.zero();
    // soln.increment(*denseCoefs[0]);
    
    pout() << "  exact value at old time " <<  imexOp.exact(time - dt) << endl;
    for (int icoef = 0; icoef < coefs.size(); ++icoef)
    {
      Real coefVal = ldfabVal(coefs[icoef]->data());
      pout() << "  coef[" << icoef << "] = " << coefVal << endl;
      Real factor = pow(theta, icoef+1) - 1;
      soln.increment(*coefs[icoef],factor);
    }
    val = ldfabVal(data);
    error = exact - val;
    denseErrs[res] = error;
    pout() << "Dense output at time " << tint << " = " << 
      val << ", error = " << error << endl;
#endif
  }

#if 0
  pout() << "Orders of convergence: " << endl;
  bool passes = true;
  // For this particular test, rates are > 4-3-2-1
  for (int res=1; res < Nres; ++res)
  {
    Real ratio = errors[res] / errors[res-1];
    Real solnrate = (-log(ratio) / log(2));
    pout() << "  soln: " << solnrate << endl;
    passes = (solnrate > (4.8-res)) && passes;
    /*
    ratio = denseErrs[res] / denseErrs[res-1];
    Real denserate = (-log(ratio) / log(2));
    pout() << "  dense output: " << denserate << endl;
    rate = min(solnrate, denserate);
    */
  }
#endif

  return (passes) ? 0 : 1;
}
