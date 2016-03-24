#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <cmath>
#include <algorithm>
#include <iostream>

#include "parstream.H"

#include "FArrayBox.H"
#include "LevelData.H"
#include "LevelDataOps.H"
#include "LayoutIterator.H"
#include "LoadBalance.H"
#include "BRMeshRefine.H"
#include "CH_Timer.H"

// ImExRK4BE test classes
#include "ImExBE.H"

#include "TestData.H"
#include "TestImExOp.H"
#include "FABView.H"
#include "FABView.H"

#include "UsingNamespace.H"

/// Global variables for handling output:
static const char* pgmname = "testLoopSolve" ;
static const char* indent = "   ";
static const char* indent2 = "      " ;

/// Prototypes:
int
testImExBE();

void
parseTestOptions(int argc ,char* argv[]) ;

Real err_tol = 2e-15;

int
main(int argc ,char* argv[])
{
#ifdef CH_MPI
  MPI_Init (&argc, &argv);
#endif
  std::cout << indent2 << "Beginning " << pgmname << " ..." << endl ;

  int status = testImExBE();

  if ( status == 0 )
    std::cout << indent << pgmname << " passed." << endl ;
  else
    std::cout << indent << pgmname << " failed with return code " << status << endl ;

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

  if (std::fabs(ldmax - ldmin) > err_tol)
  {
    std::cout << "Min/max values of error diff more than tolerance: "
      << err_tol << endl
      << "  min: " << ldmin << ", max: " << ldmax << endl;
  }

  return std::max(std::fabs(ldmax), std::fabs(ldmin));
}

int
testImExBE ()
{
  CH_TIMERS("testImExBE");
  CH_TIMER("setup",t1);
  CH_TIMER("integrate",t2);
  CH_TIMER("calc error",t3);
  CH_START(t1);

  IntVect numCells(D_DECL(32,32,20));
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
  LevelData<FArrayBox> exactLDF(dbl,1,IntVect::Zero);
  TestData soln;
  soln.aliasData(data);
  TestData exact;
  exact.aliasData(exactLDF);

  Real time = 0;
  int Nstep = 10;
  Real basedt = sqrt(.5);
  Real dt = basedt / (Real) Nstep;

  ImExBE<TestData, TestImExOp> imex;
  imex.define(soln, basedt); 
  RefCountedPtr<TestImExOp> imexOp = imex.getImExOp();
  LevelDataOps<FArrayBox> ops;
  bool passes = true;

  std::cout << "Time step: " << dt << endl;
  imex.resetDt(dt);
  // Set the initial condition
  imexOp->exact(soln, time);
  CH_STOP(t1);

  CH_START(t2);
  // Advance Nstep
  for (int step = 0; step < Nstep; ++step)
  {
    imex.advance(time, soln);
    time += dt;
  }
  CH_STOP(t2);

  CH_START(t3);
  // Calculate the error
  imexOp->exact(exact, time);
  ops.incr(exactLDF, data, -1.0);
  Real error = std::fabs(ldfabVal(exactLDF));
  std::cout << "Soln error at time " << time << " = " << error << endl;
  passes = (error <= err_tol) && passes;
  CH_STOP(t3);

  return (passes) ? 0 : 1;
}
