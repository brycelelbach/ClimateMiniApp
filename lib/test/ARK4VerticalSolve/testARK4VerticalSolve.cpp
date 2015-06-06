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
#include "FABView.H"
#include "AMRIO.H"

// ARK4 test classes
#include "ARK4v3.H"
#include "TestOpData.H"
#include "TestImExDxxOp.H"

#include "UsingNamespace.H"

/// Global variables for handling output:
static const char* pgmname = "testARK4VerticalSolve" ;
static const char* indent = "   ";
static const char* indent2 = "      " ;
static bool verbose = true ;


int
testARK4 ()
{
  // Set up the data classes
  int baseN = 8;
  // Do a convergence study, across 3 time step / resolutions
  Real basedt = 10;
  Real Nres = 4;
  // The \alpha, \beta, kx, and C in README
  TestImExDxxOp::setCoefs(.01, 0.5, 2, 1.0); 
  Vector<Real> errors(Nres,0);
  Vector<Real> denseErrs(Nres,0);
  Vector<RealVect> errorNorm(Nres);
  Vector<RealVect> denseErrNorm(Nres);

  bool denseOutput = true;
  ARK4v3<TestOpData, TestImExDxxOp> ark;
  LevelDataOps<FArrayBox> ops;
  for (int res=0; res < Nres; ++res)
  {
    Real time = 0;
    int Nstep = pow(2,res+2);
    Real dt = basedt / (Real) Nstep;
    int Nx = baseN*pow(2,res);
    pout() << "Time step: " << dt << ", Nx: " << Nx << endl;

    // Have the z dim=2 vary in size
    IntVect hiEnd = (baseN-1)*(IntVect::Unit - BASISV(0))
      + (Nx - 1)*BASISV(0);
    Vector<Box> boxes(1,Box(IntVect::Zero, hiEnd));
    Vector<int> procs(1,1);
    DisjointBoxLayout dbl(boxes,procs);

    // Allocate the dense coef data
    LevelData<FArrayBox> data(dbl,1,IntVect::Zero);
    LevelData<FArrayBox> exact(dbl,1,IntVect::Zero);
    TestOpData soln;
    soln.aliasData(data);

    // Allocate the dense coef data
    int nDenseCoefs = 4;
    Vector<TestOpData*> denseCoefs(nDenseCoefs);
    for (int icoef = 0; icoef < nDenseCoefs; ++icoef)
    {
      denseCoefs[icoef] = new TestOpData();
      denseCoefs[icoef]->define(dbl,1,IntVect::Zero);
    }

    // Define our time integrator and operator
    ark.define(soln,dt, denseOutput); 


    // Set the initial condition
    TestImExDxxOp::setExact(data, time);
    // advance nstep
    for (int step = 0; step < Nstep; ++step)
    {
      ark.advance(time, soln);
      time += dt;
    }
    // get the exact final value 
    pout() << "Solution error at final time: " << time << endl;
    TestImExDxxOp::setExact(exact, time);
    ops.incr(exact, data, -1);
    DataIterator dit = exact.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
    // FIXME - assumes just one box in 3D, need more than one
      Real vol = Nx * baseN * baseN;
      errorNorm[res][0] = exact[dit].norm(1) / vol; // L1 
      errorNorm[res][1] = exact[dit].norm(2) / sqrt(vol); // L2 
      errorNorm[res][2] = exact[dit].norm(0); // max norm
      pout() << "  Soln error = " << errorNorm[res] << endl;
    }
    
    // Compute exact dense output at some point in the last time step
    Real theta = 1 - 1/sqrt(2);
    Real tint = time - (1-theta)*dt;
    TestImExDxxOp::setExact(exact, tint);
    ark.denseOutputCoefs(denseCoefs);
    // Compute time-interpolated value in soln
    soln.copy(*denseCoefs[0]);
    for (int icoef = 1; icoef < nDenseCoefs; ++icoef)
    {
      Real factor = pow(theta, icoef);
      soln.increment(*denseCoefs[icoef],factor);
    }
    // Calculate the error
    ops.incr(exact, data, -1);
    pout() << "Dense error at time: " << tint << endl;
    for (dit.begin(); dit.ok(); ++dit)
    {
    // FIXME - assumes just one box in 3D, need more than one
      Real vol = Nx * baseN * baseN;
      denseErrNorm[res][0] = exact[dit].norm(1) / vol; // L1
      denseErrNorm[res][1] = exact[dit].norm(2) / sqrt(vol); // L2
      denseErrNorm[res][2] = exact[dit].norm(0); // max norm
      pout() << "  Dense error = " << denseErrNorm[res] << endl;
    }
  }

  // calculate convergence rates
  Real errorTol = 1.e-13; // tolerance of error in final answer (exact)
  Real goalOrder = 3.8; // order we'd like the solve to be
  Real ratio;
  RealVect rateVect;
  RealVect denseRateVect;
  pout()<<    "Convergence norms:     L1      L2    Linf" <<endl;
  for (int res=1; res < Nres; ++res)
  {
    for (int errType=0; errType<3; ++errType){
      ratio = errorNorm[res][errType] / errorNorm[res-1][errType];
      rateVect[errType] = (-log(ratio) / log(2));
      ratio = denseErrNorm[res][errType] / denseErrNorm[res-1][errType];
      denseRateVect[errType] = (-log(ratio) / log(2));
    }
    pout() << "  Solution rate: " << 
      rateVect[0]<<" "<<rateVect[1]<<" "<<rateVect[2]<< endl;
    pout() << "     Dense rate: " << 
      denseRateVect[0]<<" "<<denseRateVect[1]<<" "<<denseRateVect[2]<< endl;
  }

  // is this error / convergence rate acceptable?
  bool passedTest = true;
  if ( (errorNorm[Nres-1][2]>errorTol) && (rateVect[1] < goalOrder) )
  {
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
  pout () << indent2 << "Beginning " << pgmname << " ..." << endl ;

  int status = testARK4();

  if ( status == 0 )
    pout() << indent << pgmname << " passed." << endl ;
  else
    pout() << indent << pgmname << " failed with return code " << status << endl ;

#ifdef CH_MPI
  MPI_Finalize ();
#endif
  return status ;
}


