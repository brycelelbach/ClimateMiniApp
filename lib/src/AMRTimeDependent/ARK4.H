#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef CHOMBO_DCFC5EBE_DAAF_4F45_837E_C17A8A94C5EA
#define CHOMBO_DCFC5EBE_DAAF_4F45_837E_C17A8A94C5EA

#include <cstddef>

#include <boost/array.hpp>

#include "NamespaceHeader.H"

/// This is a more flexible, reduced memory version 4th-order additive
/// Runge-Kutta algorithm which uses the HPX backend for Chombo
/**
   This templated class encapsulates
   the fourth-order additive Runge-Kutta method 
   "ARK4(3)6L[2]SA"
   by Kennedy and Carpenter 2003 Appl. Numer. Math. 44: 139-181

   See also section 3 of Zhang, Johansen, and Colella,
   SIAM J. Sci. Comput. 34, pp. B179-B201.
*/

enum ARK4Sizes : std::size_t
{
  s_nStages = 6,
  s_nDenseCoefs = 3
};

template <class X, class Scratch, class ImExOp, bool DenseOutput>
class ARK4;

template <class X, class Scratch, class ImExOp, bool DenseOutput>
class ARK4
{
public:
  /// Runge-Kutta coefficients
  static const std::size_t  s_nStages = std::size_t(ARK4Sizes::s_nStages);
  static const std::size_t  s_nDenseCoefs = std::size_t(ARK4Sizes::s_nDenseCoefs);
  static const Real         s_aIdiag;
  static const Real         s_c[s_nStages];
  static const Real         s_aE[s_nStages][s_nStages];
  static const Real         s_aI[s_nStages][s_nStages];
  static const Real         s_b[s_nStages];
  static const Real         s_bstar[s_nDenseCoefs][s_nStages];

  template <typename TImEx>
  ARK4(TImEx&& imex, Real dt);

  template <typename... Args>
  void define(Args&&... args);

  // Advance one step.
  void advance(Real t, X& state);

private:
  void zeroDenseCoefs(std::true_type);
  void zeroDenseCoefs(std::false_type);

  void accumulateDenseCoefs(std::size_t stage, std::true_type);
  void accumulateDenseCoefs(std::size_t stage, std::false_type);

public:
  // Return current dense output coefs, 0th power first, etc.
  // NOTE: These are probably for <JU> conserved quantities
  void denseOutputCoefs(Vector<X*>& interpCoefs);

  // Reset the timestep.
  void resetDt(Real dt);

private:
  Scratch m_scratch;
  ImExOp m_opImEx;
  Real m_dt;         ///< Timestep size.
};

//==============================================

template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <typename TImEx>
ARK4<X, Scratch, ImExOp, DenseOutput>::ARK4(TImEx&& imex, Real dt)
  : m_scratch() 
  , m_opImEx(std::forward<TImEx>(imex))
  , m_dt(dt)
{}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <typename... Args>
void ARK4<X, Scratch, ImExOp, DenseOutput>::define(Args&&... args)
{
  m_scratch.define(std::forward<Args>(args)...);  
}

/*
  Advance solution state in time, t to t + dt.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::advance(Real t, X& state)
{
  auto& phi = m_scratch.phi;
  auto& kE = m_scratch.kE;
  auto& kI = m_scratch.kI;

  // Reset the dense output coefs
  zeroDenseCoefs(std::integral_constant<bool, DenseOutput>());

  // Set phi[0] := state, all others 0
  phi[0].copy(state);
  for (std::size_t stage = 1; stage < s_nStages; stage++)
    phi[stage].setVal(0.0);

  // For each stage
  for (std::size_t stage = 0; stage < s_nStages; stage++)
    {
      Real next_t = t + s_c[stage]*m_dt;
      if (stage > 0)
      {
        // Do the solve - copy rhs from phi[stage]
        phi[stage].plus(phi[0]);

        // Solve in-place for phi[stage] in
        // (I - s_aIdiag * m_dt * FI) (phi[stage]) = phi[stage].
        m_opImEx.solve(next_t, stage, s_aIdiag, phi[stage]);
      }

      // Calculate the operators for this stage
      m_opImEx.explicitOp(next_t, stage, kE, phi[stage]);
      m_opImEx.implicitOp(next_t, stage, kI, phi[stage]);

      // Add the known stage operator values to future solves 
      for (std::size_t k=stage+1; k < s_nStages; ++k)
      {
        phi[k].plus(kE, m_dt*s_aE[k][stage]);
        phi[k].plus(kI, m_dt*s_aI[k][stage]);
      }

      // Accumulate the final solution and accum registers
      state.plus(kE, m_dt*s_b[stage]);
      state.plus(kI, m_dt*s_b[stage]);

      accumulateDenseCoefs(stage, std::integral_constant<bool, DenseOutput>());
    }
}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::zeroDenseCoefs(std::true_type)
{
  for (std::size_t icoef=0; icoef < s_nDenseCoefs; ++icoef)
    m_scratch.denseCoefs[icoef].setVal(0.0);
}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::zeroDenseCoefs(std::false_type)
{}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::accumulateDenseCoefs(
    std::size_t stage
  , std::true_type
    )
{
  for (std::size_t icoef=0; icoef < s_nDenseCoefs; ++icoef)
  {
    m_scratch.denseCoefs[icoef].plus(m_scratch.kE, m_dt*s_bstar[icoef][stage]);
    m_scratch.denseCoefs[icoef].plus(m_scratch.kI, m_dt*s_bstar[icoef][stage]);
  }
} 

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::accumulateDenseCoefs(
    std::size_t stage
  , std::false_type
    )
{}

/*
  Return the coefs to interpolate solution, in terms of power of the fraction
  of time between t_old and t_new.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::denseOutputCoefs(
    Vector<X*>& interpCoefs
    )
{
  const int nCoef = s_nDenseCoefs+1;
  CH_assert(interpCoefs.size() == nCoef); 

  for (int icoef=0; icoef < nCoef; ++icoef)
    CH_assert(interpCoefs[icoef] != NULL);

  // Copy over the dense coef values

  // First coefficient is just the old state
  interpCoefs[0]->copy(m_scratch.phi[0]);
  
  // Next coefs are our dense output
  for (int icoef = 1; icoef < nCoef ; ++icoef)
  {
    interpCoefs[icoef]->copy(m_scratch.denseCoefs[icoef-1]);
  }
}

/*
  Reset the timestep.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void ARK4<X, Scratch, ImExOp, DenseOutput>::resetDt(Real dt)
{
  // Only update everything if dt has changed
  Real reltol = 1e-14;
  if (std::fabs(m_dt - dt) > m_dt*reltol)
  {
    m_dt = dt;
    m_opImEx.resetDt(m_dt);
  }
}

/*
  Static constants for ARK4
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real ARK4<X, Scratch, ImExOp, DenseOutput>::s_aIdiag = 0.25;

// Time coefficients for each stage
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
ARK4<X, Scratch, ImExOp, DenseOutput>::s_c[] = { 0.0, 0.5, 0.332, 0.62, 0.85, 1.0 };
  
// Stage coefficients - each row is for that stage 
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
ARK4<X, Scratch, ImExOp, DenseOutput>::s_aE[][ARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
  {0., 0., 0., 0., 0., 0.},
  {0.5, 0., 0., 0., 0., 0.},
  {0.221776, 0.110224, 0., 0., 0., 0.},
  {-0.04884659515311857, -0.17772065232640102, 0.8465672474795197, 0., 0., 0.},
  {-0.15541685842491548, -0.3567050098221991, 1.0587258798684427, 0.30339598837867193, 0., 0.},
  { 0.2014243506726763, 0.008742057842904185, 0.15993995707168115, 0.4038290605220775, 0.22606457389066084, 0.}
};

// Implicit stage coefficients
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
ARK4<X, Scratch, ImExOp, DenseOutput>::s_aI[][ARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
  {0., 0., 0., 0., 0., 0.},
  {0.25, 0.25, 0., 0., 0., 0.},
  {0.137776, -0.055776, 0.25, 0., 0., 0.},
  {0.14463686602698217, -0.22393190761334475, 0.4492950415863626, 0.25, 0., 0.},
  {0.09825878328356477, -0.5915442428196704, 0.8101210538282996, 0.283164405707806, 0.25, 0.},
  {0.15791629516167136, 0., 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25}
};

// Final stage coefficients
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
ARK4<X, Scratch, ImExOp, DenseOutput>::s_b[] =
  {0.15791629516167136, 0., 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25};

// Coefficients for dense output, 4th-order interpolation
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
ARK4<X, Scratch, ImExOp, DenseOutput>::s_bstar[][ARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
  {0.961753400252887, 0., 0.787405595186356, -2.74544192086633, 3.70351728061223, -1.70723435518514},
  {-1.76418754019038, 0., -0.774504669155511, 9.64023584441292, -12.544886411271, 5.44334277620397},
  {0.960350435099165, 0., 0.173858014493155, -6.21422862823726, 8.56612859966376, -3.48610842101883}
};

template <class X, bool DenseOutput>
struct ARK4DefaultScratch;

template <class X>
struct ARK4DefaultScratch<X, true>
{
  boost::array<X, ARK4Sizes::s_nStages> phi;
  boost::array<X, ARK4Sizes::s_nDenseCoefs> denseCoefs;
  X kE;
  X kI;

  void define(
    DisjointBoxLayout const& dbl
  , std::size_t comps
  , IntVect ghost
    )
  {  
    // define X
    for (std::size_t stage = 0; stage < phi.size(); ++stage)
    {
      phi[stage].define(dbl, comps, ghost);
    }
  
    // define dX
    kE.define(dbl, comps, ghost); // never exchanged.
    kI.define(dbl, comps, ghost); // never exchanged.
  
    for (std::size_t coef = 0; coef < denseCoefs.size(); ++coef)
    {
      denseCoefs[coef].define(dbl, comps, ghost); // never exchanged.
    }
  }
};
template <class X>
struct ARK4DefaultScratch<X, false>
{
  boost::array<X, ARK4Sizes::s_nStages> phi;
  X kE;
  X kI;

  void define(
    DisjointBoxLayout const& dbl
  , std::size_t comps
  , IntVect ghost
    )
  {  
    // define X
    for (std::size_t stage = 0; stage < phi.size(); ++stage)
    {
      phi[stage].define(dbl, comps, ghost);
    }
  
    // define dX
    kE.define(dbl, comps, ghost); // never exchanged.
    kI.define(dbl, comps, ghost); // never exchanged.
  }
};

#include "NamespaceFooter.H"
#endif 
