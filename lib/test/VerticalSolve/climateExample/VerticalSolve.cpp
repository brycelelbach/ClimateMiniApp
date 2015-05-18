#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif


#include "VerticalSolve.H"
#include "StencilLoopClimate.H"
#include "LapackFactorization.H"
#include "LapackWrapper.H"
#include "ClimatePhysicsVars.H"

#include "NamespaceHeader.H"

VerticalSolve::VerticalSolve()
{;}

VerticalSolve::~VerticalSolve()
{;}

void 
VerticalSolve::implicitSolve(FArrayBox& state,
                             FArrayBox& vertVel,
                             FArrayBox& stateRHS,
                             FArrayBox& vertVelRHS,
                             FArrayBox& xi,
                             FArrayBox& pEOS,
                             FArrayBox& avgJ,
                             int N,
                             Spline1DMapping* map,
                             Real gamma,
                             Real eta,
                             Real g,
                             Real height,
                             Real dt,
                             int maxIt)
{
  CH_TIMERS("operator: implicitSolve");
  CH_TIMER("allocate new memory for columns",talloc);
  CH_TIMER("pull columns of data",tpull); // TODO this ~2/3 of the time
  CH_TIMER("push columns of data",tpush); // TODO this ~2/3 of the time
  CH_TIMER("calculate Lpressure",tLp);
  CH_TIMER("calculate Lvelocity",tLv);
  CH_TIMER("Lapack ops",tLapack);
  CH_TIMER("calculate H",tH);
  CH_TIMER("post-Lapack solve updates",tupdate);
  Real dxi = 1./Real(N);

  // what are the ranges of the horizontal indices?
  Box box = state.box();
  IntVect small = box.smallEnd();
  IntVect big = box.bigEnd();

  CH_START(talloc);
  // size N: cell centers
  Real pressure[N];
  Real theta[N];
  Real pressureRHS[N];
  Real xiMat[N];
  // size N+1: vertical cell faces
  // endpoints aren't strictly necessary due to BCs but
  // this formulation minimizes the use of if statements
  Real velocity[N+1]; velocity[0] = 0.; velocity[N] = 0.;
  Real velocityRHS[N+1]; velocityRHS[0] = 0.; velocityRHS[N] = 0.;
  Real rho[N+1];
  CH_STOP(talloc);

  for (int nCol=small[1];nCol<big[1]+1;nCol++){
  for (int nRow=small[0];nRow<big[0]+1;nRow++){
    // pull columns of state, RHS, and location variables
    // writing these vectors of data
    // is actually faster than referencing the FABs directly
   
    // also convert from <JU> to <U> to second order accuracy

    CH_START(tpull);

    for (int i=0;i<N;i++){ 
      IntVect iv(nRow,nCol,i);
      xiMat[i]       = xi(iv,2); 
      // convert from <JU> to <U>
      pressure[i]    = state(iv,UPRES ) / avgJ(iv,0); 
      theta[i]       = state(iv,URHOTH) / avgJ(iv,0); 
      pressureRHS[i] = stateRHS(iv,UPRES) / avgJ(iv,0); 
    }

    for (int i=1;i<N;i++){
      IntVect iv(nRow,nCol,i);
      // boundary points are pre-assigned to be zero
      velocity[i]    = vertVel(iv,0); 
      velocityRHS[i] = vertVelRHS(iv,0); 
      //  move rho to internal cell faces to second order accuracy
      // rho at boundaries is never actually used
      rho[i] = 0.5*( state(IntVect(nRow,nCol,i-1),URHO) / avgJ(IntVect(nRow,nCol,i-1),0)
                   + state(IntVect(nRow,nCol,i),URHO)  / avgJ(IntVect(nRow,nCol,i),0) ); 
    }

    CH_STOP(tpull);
  
    // loop over iterations of Newton solver
    int l; // iteration number
    for (l=0;l<maxIt;l++){ 
      // calculate -L_pressure
      // this is written up in CubedSphereShell/doc/dycore3D/design3D.tex
      CH_START(tLp);
      Vector<Real> minusLPressure(N);
      for (int i=0;i<N;i++){
        Real drdxi = map->getDerivative( xiMat[i] ); drdxi*=height; // at cell center
        Real dr = dxi*drdxi;
        Real wDiff = velocity[i+1] - velocity[i];
        Real dwdr = wDiff/dr;
        Real pEOSVal = pEOS(IntVect(nRow,nCol,i),0);
        Real LIPressure = -gamma*pressure[i]*dwdr + eta*(pEOSVal - pressure[i]);

        // L_p = pressure - dt*L_I_pressure - rhs_pressure
        minusLPressure[i] = pressure[i] + 
                            -dt*LIPressure +
                            -pressureRHS[i];
        minusLPressure[i] *= -1.; // since we're interested in -L(S)
      }
      CH_STOP(tLp);

      // calculate -L_velocity
      // this is written up in CubedSphereShell/doc/dycore3D/design3D.tex
      CH_START(tLv);
      Vector<Real> minusLVelocity(N-1); // only update velocity on internal faces
      for (int i=0;i<N-1;i++){
        Real pDiff = pressure[i+1] - pressure[i];
        Real drdxi = map->getDerivative( xiMat[i+1]-0.5*dxi ); drdxi *= height; // at cell bottom
        Real dr = dxi*drdxi;
        Real LIVelocity = -(1./rho[i+1])*pDiff/dr - g;

        // L_v = velocity - dt*L_I_velocity - rhs_velocity
        minusLVelocity[i] = velocity[i+1]  +
                           -dt*LIVelocity +
                           -velocityRHS[i+1];
        minusLVelocity[i] *= -1.; // since we're interested in -L(S)
      }
      CH_STOP(tLv);

      // calculate A = dL_p/dp
      // calculate B = dL_p/dw
      // calculate C = dL_w/dp
      // D = dW_w/dw = I, so don't need to calculate it

      // solve for change in pressure from
      // (A - BD^-1C) Delta_p = -L_pressure - BD^-1(-L_velocity)
      CH_START(tLapack);
      LapackFactorization H; 
      H.define(N,1,1);
      CH_STOP(tLapack);
      CH_START(tH);
      Real* L = new Real[N];
      for (int i=0;i<N;i++){

        Real wDiff = velocity[i+1] - velocity[i];

        Real drdxiCen = map->getDerivative( xiMat[i] ); drdxiCen *= height;
        Real drCen = dxi*drdxiCen; // dr for cell centers

        Real drdxiFaceP = map->getDerivative( xiMat[i]+0.5*dxi ); drdxiFaceP *= height;
        Real drFaceP = dxi*drdxiFaceP; // dr for face centers, upper face 

        Real drdxiFaceM = map->getDerivative( xiMat[i]-0.5*dxi ); drdxiFaceM *= height;
        Real drFaceM = dxi*drdxiFaceM; // dr for face centers, lower face 

        Real dLdw  = dt*gamma*pressure[i]/drCen; // dLp/dw, makes B
        Real rhoInv = dt*(1./rho[i  ])/drFaceM; // dLw/dp, makes C 
        Real rhoInvP= dt*(1./rho[i+1])/drFaceP; // dLw/dp, makes C
        Real dLdwRhoInv = dLdw*rhoInv; // part of B*C, useful for defining H
        Real dLdwRhoInvP= dLdw*rhoInvP; // part of B*C, useful for defining H

        // define H = A - BC
        // and
        // define L = -L_p - B(-L_v)
        H(i,i) = 1. + dt*gamma*(1./drCen)*wDiff + dt*eta; // contribution from A = dL_p/dp
        L[i] = minusLPressure[i]; // contribution from -L_p
        // contribution to H from B*C and to L from B*-L_v
        // this is written up in CubedSphereShell/doc/dycore3D/design3D.tex
        if (i>0 and i<N-1){ // interior rows
          H(i,i)   -= -dLdwRhoInv + -dLdwRhoInvP;
          H(i,i-1) -=  dLdwRhoInv; 
          H(i,i+1) -=  dLdwRhoInvP;
          L[i] += -( -dLdw*minusLVelocity[i-1] + dLdw*minusLVelocity[i] );
        } else if (i==0) { // first row
          H(i,i  ) -= -dLdwRhoInvP; 
          H(i,i+1) -=  dLdwRhoInvP;
          L[i] += -( dLdw*minusLVelocity[0]);
        } else if (i==N-1) { // last row
          H(i,i  ) -= -dLdwRhoInv ;
          H(i,i-1) -=  dLdwRhoInv ;
          L[i] += -( -dLdw*minusLVelocity[N-2] );
        } // if-else to find end points

      } // loop over rows of H and L
      CH_STOP(tH);

      // solve H(Delta S) = L for (Delta S)
      CH_START(tLapack);
      LapackWrapper::factorBandMatrix(H);
      LapackWrapper::solveBandMatrix(H,L);
      CH_STOP(tLapack);
      // now L = \Delta p

      CH_START(tupdate);
      // update pressure state
      for (int i=0;i<N;i++){
        pressure[i] += L[i];
      }

      // update velocity state using
      // \Delta w = D^-1(-L_velocity - CDelta_p)
      // this is written up in CubedSphereShell/doc/dycore3D/design3D.tex
      for (int i=0;i<N-1;i++){
        Real drdxi = map->getDerivative( xiMat[i+1]-0.5*dxi ); drdxi *= height; // at cell bottom
        Real dr = dxi*drdxi;
        Real dLwdp = dt*(1./rho[i+1])/dr;
        Real deltaW = minusLVelocity[i] - (-dLwdp*L[i] + dLwdp*L[i+1]); 
        
        velocity[i+1] += deltaW; 
      }
      CH_STOP(tupdate);

      // clean up
      delete L;
    } // loop over Newton iterations

  CH_START(tpush);
  // put solution state back into FAB
  for (int i=0;i<N;i++){ 
    // also convert <U> to <JU>
    state(IntVect(nRow,nCol,i),UPRES) = pressure[i]*avgJ(IntVect(nRow,nCol,i),0);
  }
  for (int i=1;i<N;i++){
    vertVel(IntVect(nRow,nCol,i),0) = velocity[i];
  }
  CH_STOP(tpush);

  }} // loop over columns
}
