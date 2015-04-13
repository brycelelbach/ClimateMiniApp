#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _ARK4V3_H_
#define _ARK4V3_H_


#include "NamespaceHeader.H"

/// This is a more flexible, reduced memory version 4th-order additive
/// Runge-Kutta algorithm
/**
   This templated class encapsulates
   the fourth-order additive Runge-Kutta method 
   "ARK4(3)6L[2]SA"
   by Kennedy and Carpenter 2003 Appl. Numer. Math. 44: 139-181

   See also section 3 of Zhang, Johansen, and Colella,
   SIAM J. Sci. Comput. 34, pp. B179-B201.
*/

template <class X, class IMEXOP>
class ARK4
{
public:
  ARK4(
    IMEXOP&& imex
  , const X& state
  , Real dt
  , bool denseOutput = false
    );

  // Advance one step.
  void advance(
    Real t
  , X& state
    );

  // Return current dense output coefs, 0th power first, etc.
  // NOTE: These are probably for <JU> conserved quantities
  void denseOutputCoefs(Vector<X*>& interpCoefs);

  // Reset the timestep.
  void resetDt(Real dt);

  // Access to the operators if we're defined already
  // Caller is responsible for making sure they're in a good state 
  // (for example, like have resetDt() called, etc.
  IMEXOP& getImExOp();

  /// Runge-Kutta coefficients
  static const int  s_nStages = 6;
  static const Real s_aIdiag;
  static const Real s_c[s_nStages];
  static const Real s_aE[s_nStages][s_nStages];
  static const Real s_aI[s_nStages][s_nStages];
  static const Real s_b[s_nStages];
  static const int  s_nDenseCoefs = 3;
  static const Real s_bstar[s_nDenseCoefs][s_nStages];

private:
  bool m_denseOutput;               ///< True if dense output is enabled.
  Real m_dt;                        ///< Timestep size.

  X m_phi[s_nStages];
  X m_denseCoefs[s_nDenseCoefs];
  X m_kE;
  X m_kI;

  IMEXOP m_opImEx;
};

//==============================================

template <class X, class IMEXOP>
ARK4<X, IMEXOP>::ARK4(
    IMEXOP&& imex
  , X const& state
  , Real dt
  , bool denseOutput
    )
  : m_opImEx(imex)
{
  m_dt = dt;
  m_denseOutput = denseOutput;

  // define X
  for (int stage = 0; stage < s_nStages; stage++)
    m_phi[stage].define(state);

  // define dX
  m_kE.define(state);
  m_kI.define(state);

  // if dense output is requested, need more storage
  for (int coef = 0; m_denseOutput && (coef < s_nDenseCoefs); coef++)
    m_denseCoefs[coef].define(state);
}

/*
  Get a reference to the implicit-explicit operator
 */
template <class X, class IMEXOP>
IMEXOP& 
ARK4<X, IMEXOP>::getImExOp()
{
  return m_opImEx;
}

/*
  Reset the timestep.
 */
template <class X, class IMEXOP>
void ARK4<X, IMEXOP>::resetDt(Real dt)
{
  // Only update everything if dt has changed
  Real reltol = 1e-14;
  if (fabs(m_dt - dt) > m_dt*reltol)
  {
    m_dt = dt;
    m_opImEx.resetDt(m_dt);
  }
}

/*
  Advance solution state in time, t to t + dt.
 */
template <class X, class IMEXOP>
void ARK4<X, IMEXOP>::advance(Real t, X& state)
{
  // Reset the dense output coefs
  if (m_denseOutput)
  {
    for (int icoef=0; icoef < s_nDenseCoefs; ++icoef)
      m_denseCoefs[icoef].zero();
  }

  // Set m_phi[0] := state, all others 0
  m_phi[0].copy(state);
  for (int stage = 1; stage < s_nStages; stage++)
    m_phi[stage].zero();

  // Set the output to zero
  // state.zero();

  // For each stage
  for (int stage = 0; stage < s_nStages; stage++)
    {
      Real next_t = t + s_c[stage]*m_dt;
      if (stage > 0)
      {
        // Do the solve - copy rhs from phi[stage]
        m_phi[stage].increment(m_phi[0]);
   
        m_phi[stage].exchange(); 

        // Solve in-place for m_phi[stage] in
        // (I - s_aIdiag * m_dt * FI) (m_phi[stage]) = m_phi[stage].
        m_opImEx.solve(m_phi[stage], next_t, s_aIdiag);
      }

      m_phi[stage].exchange();

      // Calculate the operators for this stage
      m_opImEx.explicitOp(m_kE, m_phi[stage], next_t);
      m_opImEx.implicitOp(m_kI, m_phi[stage], next_t);

      // Add the known stage operator values to future solve rhs in phi[stage]
      for (int k=stage+1; k < s_nStages; ++k)
      {
        m_phi[k].increment(m_kE, m_dt*s_aE[k][stage]);
        m_phi[k].increment(m_kI, m_dt*s_aI[k][stage]);
      }

      // Accumulate the final solution and accum registers
      state.increment(m_kE, m_dt*s_b[stage]);
      state.increment(m_kI, m_dt*s_b[stage]);

      if (m_denseOutput)
      {
        for (int icoef=0; icoef < s_nDenseCoefs; ++icoef)
        {
          m_denseCoefs[icoef].increment(m_kE, m_dt*s_bstar[icoef][stage]);
          m_denseCoefs[icoef].increment(m_kI, m_dt*s_bstar[icoef][stage]);
        }
      }
    }
}

/*
  Return the coefs to interpolate solution, in terms of power of the fraction
  of time between t_old and t_new.
 */
template <class X, class IMEXOP>
void ARK4<X, IMEXOP>::denseOutputCoefs(Vector<X*>& interpCoefs)
{
  const int nCoef = s_nDenseCoefs+1;
  CH_assert(interpCoefs.size() == nCoef); 

  for (int icoef=0; icoef < nCoef; ++icoef)
    CH_assert(interpCoefs[icoef] != NULL);

  // Copy over the dense coef values

  // First coeficient is just the old state
  interpCoefs[0]->copy(m_phi[0]);
  
  // Next coefs are our dense output
  for (int icoef = 1; icoef < nCoef ; ++icoef)
  {
    interpCoefs[icoef]->copy(m_denseCoefs[icoef-1]);
  }
}


/*
  Static constants for ARK4
 */

template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_aIdiag = 0.25;

// Time coefficients for each stage
template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_c[] = { 0.0, 0.5, 0.332, 0.62, 0.85, 1.0 };
  
// Stage coefficients - each row is for that stage 
template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_aE[][ARK4<X, IMEXOP>::s_nStages] = {
  {0., 0., 0., 0., 0., 0.},
  {0.5, 0., 0., 0., 0., 0.},
  {0.221776, 0.110224, 0., 0., 0., 0.},
  {-0.04884659515311857, -0.17772065232640102, 0.8465672474795197, 0., 0., 0.},
  {-0.15541685842491548, -0.3567050098221991, 1.0587258798684427, 0.30339598837867193, 0., 0.},
  { 0.2014243506726763, 0.008742057842904185, 0.15993995707168115, 0.4038290605220775, 0.22606457389066084, 0.}
};

// Implicit stage coefficients
template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_aI[][ARK4<X, IMEXOP>::s_nStages] = {
  {0., 0., 0., 0., 0., 0.},
  {0.25, 0.25, 0., 0., 0., 0.},
  {0.137776, -0.055776, 0.25, 0., 0., 0.},
  {0.14463686602698217, -0.22393190761334475, 0.4492950415863626, 0.25, 0., 0.},
  {0.09825878328356477, -0.5915442428196704, 0.8101210538282996, 0.283164405707806, 0.25, 0.},
  {0.15791629516167136, 0., 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25}
};

// Final stage coefficients
template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_b[] =
  {0.15791629516167136, 0., 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25};

// Coefficients for dense ouput, 4th-order interpolation
template <class X, class IMEXOP>
const Real ARK4<X, IMEXOP>::s_bstar[][ARK4<X, IMEXOP>::s_nStages] = {
  {0.961753400252887, 0., 0.787405595186356, -2.74544192086633, 3.70351728061223, -1.70723435518514},
  {-1.76418754019038, 0., -0.774504669155511, 9.64023584441292, -12.544886411271, 5.44334277620397},
  {0.960350435099165, 0., 0.173858014493155, -6.21422862823726, 8.56612859966376, -3.48610842101883}
};

#include "NamespaceFooter.H"
#endif 