#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _ImExRK4BE_H_
#define _ImExRK4BE_H_

#include "CH_Timer.H"
#include "TestImExOp.H"
#include "StencilLoopOps.H"

#include "NamespaceHeader.H"

/// This is an Implicit-Explicit integrator, with
/// 4th-order explicit Runge-Kutta called before
/// backward Euler implicit solve and a final
/// forward Euler explicit source term solve

template <class Soln, class Rhs, class IMEXOP>
class ImExRK4BE
{
public:

  ImExRK4BE<Soln, Rhs, IMEXOP>() { m_isDefined = false; }

  // This must be called first.
  void define(const Soln& a_state, Real a_dt, bool a_denseOutput = false);

  // Advance one step.
  void advance(Real a_time, Soln& a_state);

  // Return current dense output coefs to modify from final state, 
  // 1st power first, etc.
  Vector<Rhs*> denseOutputCoefs();

  // Reset the timestep.
  void resetDt(Real a_dt);

  // Access to the operators if we're defined already
  // Caller is responsible for making sure they're in a good state 
  // (for example, like have resetDt() called, etc.
  RefCountedPtr<IMEXOP> getImExOp() const;

  // Access to the Soln, Rhs classes via pointers
  Vector<Soln*>& getSolnPointers();
  Vector<Rhs*>& getRhsPointers();

  bool isDefined() const { return m_isDefined; }

  bool hasDenseOutput() const { return m_hasDenseOutput; }

  /// Runge-Kutta coefficients
  static const int  s_nStages = 4;
  static const Real s_c[s_nStages];
  static const Real s_aE[s_nStages];
  static const Real s_b[s_nStages];
  static const int  s_nDenseCoefs = 3;
  static const Real s_bstarE[s_nDenseCoefs][s_nStages];

protected:
  bool m_isDefined;
  bool m_denseOutput;
  bool m_hasDenseOutput; // If there is dense output data to interpolate
  Real m_dt;
  Real m_time;
  Soln m_tmp;
  Soln m_stage;
  Soln m_final;
  Rhs m_denseCoefs[s_nDenseCoefs];
  Rhs m_kE;
  Rhs m_kI;
  RefCountedPtr<IMEXOP> m_opImExPtr;

private:

};

//==============================================

template <class Soln, class Rhs, class IMEXOP>
void ImExRK4BE<Soln, Rhs, IMEXOP>::define(
    const Soln& a_state, Real a_dt, bool a_denseOutput)
{
  CH_TIME("ImExRK4BE::define");

  m_dt = a_dt;
  m_denseOutput = a_denseOutput;
  m_tmp.define(a_state);
  m_stage.define(a_state);
  m_final.define(a_state);

  // define dX
  m_kE.define(a_state);
  m_kI.define(a_state);

  // define opImEx
  if (m_opImExPtr.isNull())
    m_opImExPtr = RefCountedPtr<IMEXOP>(new IMEXOP());
  m_opImExPtr->define(a_state, m_dt, 1.0);

  // if dense output is requested, need storage for that
  for (int coef = 0; m_denseOutput && (coef < s_nDenseCoefs); coef++)
    m_denseCoefs[coef].define(a_state);

  m_hasDenseOutput = false;
  m_isDefined = true;
}

/*
  Get a reference to the implicit-explicit operator
 */
template <class Soln, class Rhs, class IMEXOP>
RefCountedPtr<IMEXOP> 
ImExRK4BE<Soln, Rhs, IMEXOP>::getImExOp() const
{
  return m_opImExPtr;
}

/*
  Reset the timestep.
 */
template <class Soln, class Rhs, class IMEXOP>
void ImExRK4BE<Soln, Rhs, IMEXOP>::resetDt(Real a_dt)
{
  CH_assert(isDefined());

  // Only update everything if dt has changed
  Real reltol = 1e-12;
  if (fabs(m_dt - a_dt) > m_dt*reltol)
  {
    m_dt = a_dt;
    m_hasDenseOutput = false;
    m_opImExPtr->resetDt(m_dt);
  }
}

/*
  Advance solution a_state in time, a_time to a_time + a_dt.
 */
template <class Soln, class Rhs, class IMEXOP>
void ImExRK4BE<Soln, Rhs, IMEXOP>::advance(Real a_time, Soln& a_state)
{
  CH_TIMERS("ImExRK4BE::advance");
  CH_TIMER("ImExRK4BE::advance explicit", t1);
  CH_TIMER("ImExRK4BE::advance implicit", t2);
  CH_TIMER("ImExRK4BE::advance source", t3);
  CH_assert(isDefined());

  // Reset the dense output coefs
/*
  if (m_denseOutput)
  {
    m_hasDenseOutput = false;
    for (int icoef=0; icoef < s_nDenseCoefs; ++icoef)
      m_denseCoefs[icoef].zero();
  }

  CH_START(t1);
  {
    // stage 0
    int stage = 0;
    Real t = a_time + s_c[stage]*m_dt;
    Real finalScale = m_dt*s_b[stage];
    a_state.exchange();
    Real nextScale = m_dt*s_aE[stage+1];
    // Initialize the accumulate pointers and scales
    RKAccum<2> rk4s0;
    rk4s0.scale[0] = nextScale;
    rk4s0.scale[1] = finalScale;
    rk4s0.accum[0] = &m_stage;
    rk4s0.accum[1] = &m_final;

    DataIterator dit = a_state.data().dataIterator();
    const Vector<pair<DataIndex,Box> > vecTiles = a_state.getTiles(dit);
    for (int tix=0; tix < vecTiles.size(); ++tix) 
    {
      m_final.copy(vecTiles[tix], a_state);
      m_stage.copy(vecTiles[tix], a_state);
      explicitOp<2>(vecTiles[tix], m_kE, t, a_state, rk4s0);
    }
    
    // stage 1
    stage = 1;
    t = a_time + s_c[stage]*m_dt;
    nextScale = m_dt*s_aE[stage+1];
    finalScale = m_dt*s_b[stage];
    // Initialize the accumulate pointers and scales
    RKAccum<2> rk4s1;
    rk4s1.scale[0] = nextScale;
    rk4s1.scale[1] = finalScale;
    rk4s1.accum[0] = &m_tmp;
    rk4s1.accum[1] = &m_final;
    m_stage.exchange();
    for (int tix=0; tix < vecTiles.size(); ++tix) 
    {
      m_tmp.copy(vecTiles[tix], a_state);
      explicitOp<2>(vecTiles[tix], m_kE, t, m_stage, rk4s1);
    }
    
    // stage 2
    stage = 2;
    t = a_time + s_c[stage]*m_dt;
    nextScale = m_dt*s_aE[stage+1];
    finalScale = m_dt*s_b[stage];
    // Initialize the accumulate pointers and scales
    RKAccum<2> rk4s2;
    rk4s2.scale[0] = nextScale;
    rk4s2.scale[1] = finalScale;
    rk4s2.accum[0] = &m_stage;
    rk4s2.accum[1] = &m_final;
    m_tmp.exchange();
    for (int tix=0; tix < vecTiles.size(); ++tix) 
    {
      m_stage.copy(vecTiles[tix], a_state);
      explicitOp<2>(vecTiles[tix], m_kE, t, m_tmp, rk4s2);
    }

    // stage 3
    stage = 3;
    t = a_time + s_c[stage]*m_dt;
    nextScale = 0; // No next stage update
    finalScale = m_dt*s_b[stage];
    // Initialize the accumulate pointers and scales
    RKAccum<1> rk4s3;
    rk4s3.scale[0] = finalScale;
    rk4s3.accum[0] = &m_final;
    m_stage.exchange();
    for (int tix=0; tix < vecTiles.size(); ++tix) 
    {
      explicitOp<1>(vecTiles[tix], m_kE, t, m_stage, rk4s3);
      a_state.copy(vecTiles[tix], m_final);
    }
  }

  // Calculate the final stage update for the explicit operator
  // m_kE.copy(a_state);
  // m_kE.increment(m_kI, 1); // m_kE is tmp state for solve, phi^n + dt*expOp
  // a_state.increment(m_kI, 1); // m_kE is tmp state for solve, phi^n + dt*expOp
  // final from the RK4 integrator   
  // a_state.copy(m_final); // TODO - loop copy
  CH_STOP(t1);
*/

  CH_START(t2);
  // Implicit backward Euler solve
  Real t = a_time + m_dt;
  // m_kI.copy(a_state);
  // implicitSolve(a_state, m_kI, t, m_dt);
  DataIterator dit = a_state.data().dataIterator();
  const Vector<pair<DataIndex,Box> > vecTiles = a_state.getTiles(dit);
  for (int tix=0; tix < vecTiles.size(); ++tix) 
  {
    m_kI.copy(vecTiles[tix], a_state);
    implicitSolve(vecTiles[tix], a_state, m_kI, t, m_dt);
  }
  CH_STOP(t2);

/*
  if (m_denseOutput)
  {
    // Only linear terms are updated with m_kI
    m_denseCoefs[0].increment(m_kI, m_dt);

    // And for the operator-split source term (linear)
    m_denseCoefs[0].increment(m_kE, m_dt);
  }

  m_hasDenseOutput = m_denseOutput;
*/
  m_time = a_time;
}


/*
  Return the coefs to interpolate solution, in terms of power of the fraction
  of time between t_old and t_new.
 */
template <class Soln, class Rhs, class IMEXOP>
Vector<Rhs*> ImExRK4BE<Soln, Rhs, IMEXOP>::denseOutputCoefs()
{
  MayDay::Abort("Dense output not implemented in ImExRK4BE::denseOutputCoefs");

  const int nCoef = s_nDenseCoefs;
  CH_assert(m_hasDenseOutput);
  Vector<Rhs*> vecCoefs(nCoef); 
  
/*
  // Next coefs are our dense output
  for (int icoef = 0; icoef < nCoef ; ++icoef)
  {
    // LevelData<FArrayBox>& coef = m_denseCoefs[icoef].data();
    // for (dit.begin(); dit.ok(); ++dit)
    //   pout() << "  Coef[" << icoef+1 << "] = " << coef[dit].min() << endl;
    vecCoefs[icoef] = m_rhsPointers[icoef + DENSE1];
  }
*/

  return vecCoefs;
}

/*
  Static constants for RK4
 */

// Time coefficients for each stage
template <class Soln, class Rhs, class IMEXOP>
const Real ImExRK4BE<Soln, Rhs, IMEXOP>::
s_c[] = { 0.0, 0.5, 0.5, 1.0 };
  
// Stage coefficients - each row is for that stage 
template <class Soln, class Rhs, class IMEXOP>
const Real ImExRK4BE<Soln, Rhs, IMEXOP>::
s_aE[ImExRK4BE<Soln, Rhs, IMEXOP>::s_nStages] = {0., 0.5, 0.5, 1.};

// Final stage coefficients
template <class Soln, class Rhs, class IMEXOP>
const Real ImExRK4BE<Soln, Rhs, IMEXOP>::
s_b[] =
  {0.16666666666666667, 0.33333333333333333, 0.33333333333333333, 0.16666666666666667};

// Coefficients for dense ouput, 4th-order interpolation in RK4
template <class Soln, class Rhs, class IMEXOP>
const Real ImExRK4BE<Soln, Rhs, IMEXOP>::
s_bstarE[][ImExRK4BE<Soln, Rhs, IMEXOP>::s_nStages] = {
  {1.0, 0., 0., 0.},
  {-1.5, 1.0, 1.0, -0.5},
  {0.66666666666666667, -0.66666666666666667, -0.66666666666666667, 0.66666666666666667}
};

#include "NamespaceFooter.H"
#endif 
