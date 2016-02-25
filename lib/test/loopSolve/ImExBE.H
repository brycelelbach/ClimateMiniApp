#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _ImExBE_H_
#define _ImExBE_H_

#include "CH_Timer.H"
#include "TestImExOp.H"

#include "NamespaceHeader.H"

/// This is an Implicit backward Euler integrator

template <class Soln, class IMEXOP>
class ImExBE
{
public:

  ImExBE<Soln, IMEXOP>() { m_isDefined = false; }

  // This must be called first.
  void define(const Soln& a_state, Real a_dt);

  // Advance one step.
  void advance(Real a_time, Soln& a_state);

  // Reset the timestep.
  void resetDt(Real a_dt);

  // Access to the operators if we're defined already
  // Caller is responsible for making sure they're in a good state 
  // (for example, like have resetDt() called, etc.
  RefCountedPtr<IMEXOP> getImExOp() const;

  bool isDefined() const { return m_isDefined; }

protected:
  bool m_isDefined;
  Real m_dt;
  Real m_time;
  Soln m_kI;
  RefCountedPtr<IMEXOP> m_opImExPtr;

private:

};

//==============================================

template <class Soln, class IMEXOP>
void ImExBE<Soln, IMEXOP>::define(
    const Soln& a_state, Real a_dt)
{
  CH_TIME("ImExBE::define");

  m_dt = a_dt;

  // define dX
  m_kI.define(a_state);

  // define opImEx
  if (m_opImExPtr.isNull())
    m_opImExPtr = RefCountedPtr<IMEXOP>(new IMEXOP());
  m_opImExPtr->define(a_state, m_dt, 1.0);

  m_isDefined = true;
}

/*
  Get a reference to the implicit-explicit operator
 */
template <class Soln, class IMEXOP>
RefCountedPtr<IMEXOP> 
ImExBE<Soln, IMEXOP>::getImExOp() const
{
  return m_opImExPtr;
}

/*
  Reset the timestep.
 */
template <class Soln, class IMEXOP>
void ImExBE<Soln, IMEXOP>::resetDt(Real a_dt)
{
  CH_assert(isDefined());

  // Only update everything if dt has changed
  Real reltol = 1e-12;
  if (fabs(m_dt - a_dt) > m_dt*reltol)
  {
    m_dt = a_dt;
    m_opImExPtr->resetDt(m_dt);
  }
}

/*
  Advance solution a_state in time, a_time to a_time + a_dt.
 */
template <class Soln, class IMEXOP>
void ImExBE<Soln, IMEXOP>::advance(Real a_time, Soln& a_state)
{
  CH_TIME("ImExBE::advance");
  CH_assert(isDefined());

  // Implicit backward Euler solve
  Real t = a_time + m_dt;
  DataIterator dit = a_state.data().dataIterator();
  const Vector<pair<DataIndex,Box> > vecTiles = a_state.getTiles(dit);
  for (int tix=0; tix < vecTiles.size(); ++tix) 
  {
    m_kI.copy(vecTiles[tix], a_state);
    m_opImExPtr->implicitSolve(vecTiles[tix], a_state, m_kI, t, m_dt);
  }

  m_time = a_time;
}

#include "NamespaceFooter.H"
#endif 
