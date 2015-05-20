#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <iostream>
#include "parstream.H"
#include "ParmParse.H"
#include "CH_Timer.H"
#include "FArrayBox.H"
#include "LayoutIterator.H"
#include "LoadBalance.H"
#include "BRMeshRefine.H"
#include "CONSTANTS.H"
#include "LevelData.H"

// test classes
#include "LapackFactorization.H"
#include "LapackWrapper.H"
#include "TestOperator.H"

#include "FABView.H"

#include "UsingNamespace.H"

/// Global variables for handling output:
static const char* pgmname = "testVerticalSolve" ;
static const char* indent = "   ";
static const char* indent2 = "      " ;

/// Main test loop
int
testVerticalSolveConvergence()
{
    CH_TIMERS("testVerticalSolveConvergence");

    bool passedTest = true;

    // parameters that should probably be in an input file
    int minN = 10;     // minimum number of vertical cells
    int nRes = 4;     // number of resolutions to test
    Real minDt = 1.0;    // timestep
    Real errorTol = 1.e-13; // tolerance of error in final answer (exact)
    Real goalOrder = 1.8; // order we'd like the solve to be

    // store errors to calculate convergence rates
    Vector<RealVect> errorNorm(nRes);

    // loop over resolutions to test
    for (int res=0;res<nRes;res++){

      int N = minN*pow(2,res);
      Box b(IntVect::Zero, (N-1)*IntVect::Unit);
      FArrayBox soln(b,1);
      FArrayBox rhs(b,1);

      Real dx = 1.0/(Real) N;
      // Real dt = minDt * dx; // temporal error for 1 step is O(dt^2)
      Real dt = minDt; // temporal error for 1 step is O(dt^2)
      Real coef = dt;
      int kx = 2;

      // Initialize the rhs to exact soln
      Real initCoef = 1;
      TestOperator::setExact(rhs, kx, dx, initCoef);

      // Build the Laplacian
      // band_matrix L;
      // TestOperator::build2ndOrderOperator(L, coef, dx, N);
 
      // Build the Helmholtz matrix and solve
      band_matrix H;
      TestOperator::build4thOrderSolver(H, coef, dx, N);
      TestOperator::implicitSolve(soln, rhs, H);

      // Just make sure the residual is zero
      /*
      // Soln is initial soln value
      TestOperator::setExact(soln, kx, dx, initCoef); 
      // apply coef D_{xx} to it
      Real alpha = coef;
      Real beta = 0.0;
      TestOperator::applyBandedMatrix(soln, rhs, L, alpha, beta);
      // pout() << "N: " << N <<" residual = "<< rhs.norm(0) << endl;
      */
 
      // Calculate the error versus exact solution
      Real exactcoef = 1.0 / (1.0 + coef * pow(M_PI*(Real) kx,2));
      // Real exactcoef = -coef * pow(M_PI*(Real) kx,2);
      TestOperator::setExact(rhs, kx, dx, exactcoef);
      rhs.minus(soln);

      // Output soln error norms
      errorNorm[res][0] = rhs.norm(1) / pow((Real) N, 3); // L1 norm
      errorNorm[res][1] = rhs.norm(2) / pow((Real) N, 1.5); // L2 norm
      errorNorm[res][2] = rhs.norm(0); // max norm
      pout()<<"N: "<<N<<" soln error = "<<errorNorm[res]<<endl;

    } // loop over resolutions

    // calculate convergence rates
    Real ratio;
    RealVect rate;
    pout()<<    "                        L1     L2     Linf" <<endl;
    for (int res=1; res < nRes; ++res)
    {
      for (int errType=0; errType<3; ++errType){
        ratio = errorNorm[res][errType] / errorNorm[res-1][errType];
        rate[errType] = (-log(ratio) / log(2));
      }
      pout() << "Convergence rate: " << rate[0]<<" "<<rate[1]<<" "<<rate[2]<< endl;
    }
  
    // is this error / convergence rate acceptable?
    // if the solution isn't exact and the convergence rate is too low, then not acceptable
    if ( (errorNorm[nRes-1][2]>errorTol) && (rate[1] < goalOrder) ){
      passedTest = false;
      pout()<<"Failed convergence test!"<<endl;
    }

    if (passedTest) pout()<<"Passed convergence test!"<<endl;

    return !passedTest;
}


int
main(int argc ,char* argv[])
{
#ifdef CH_MPI
  MPI_Init (&argc, &argv);
#endif

  // Check for an input file
  char* inFile = NULL;

  /*
  if (argc > 1)
  {
    inFile = argv[1];
  }
  else
  {
    pout() << "Usage:  ...ex <inputfile>" << endl;
    pout() << "No input file specified" << endl;
    return -1;
  }

  // Parse the input file
  ParmParse pp(argc-2, argv+2, NULL, inFile);
  */

  pout () << indent2 << "Beginning " << pgmname << " ..." << endl ;

  int status = 0;

  // test that the linear vertical solve converges at fourth order
  // against an exact solution
  status += testVerticalSolveConvergence();

  pout()<<"-------------------------------------------"<<endl;
  if ( status == 0 )
    pout() << indent << pgmname << " passed." << endl ;
  else
    pout() << indent << pgmname << " failed with return code " << status << endl ;

  CH_TIMER_REPORT();
#ifdef CH_MPI
  MPI_Finalize ();
#endif
  return status ;
}




