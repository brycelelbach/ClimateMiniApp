#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef CHOMBO_6415B1D5_CF9D_451C_AC98_AE1A952E2B9B
#define CHOMBO_6415B1D5_CF9D_451C_AC98_AE1A952E2B9B

#include <cstddef>

#include <boost/array.hpp>

#include "ARK4Sizes.H"

#include "NamespaceHeader.H"

/// This is a more flexible, reduced memory version 4th-order additive
/// Runge-Kutta algorithm which uses the HPX backend for Chombo
/**
   This templated class encapsulates
   the fourth-order additive Runge-Kutta method 
   "OMPARK4(3)6L[2]SA"
   by Kennedy and Carpenter 2003 Appl. Numer. Math. 44: 139-181

   See also section 3 of Zhang, Johansen, and Colella,
   SIAM J. Sci. Comput. 34, pp. B179-B201.
*/

template <class X, class Scratch, class ImExOp, bool DenseOutput>
class OMPARK4;

template <class X, class Scratch, class ImExOp, bool DenseOutput>
class OMPARK4
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
  OMPARK4(TImEx&& imex, Real dt);

  template <typename... Args>
  void define(Args&&... args);

  // Advance one step.
  void advance(Real t, X& state);

  template <std::size_t Stage>
  // Advance one stage.
  void advanceStage(Real t, X& state);

private:
  void zeroDenseCoefs(std::size_t ibox, std::true_type);
  void zeroDenseCoefs(std::size_t ibox, std::false_type);

  template <std::size_t Stage>
  void accumulateDenseCoefs(std::size_t ibox, std::true_type);

  template <std::size_t Stage>
  void accumulateDenseCoefs(std::size_t ibox, std::false_type);

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
OMPARK4<X, Scratch, ImExOp, DenseOutput>::OMPARK4(TImEx&& imex, Real dt)
  : m_scratch() 
  , m_opImEx(std::forward<TImEx>(imex))
  , m_dt(dt)
{}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <typename... Args>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::define(Args&&... args)
{
  m_scratch.define(std::forward<Args>(args)...);  
}

/*
  Advance solution state in time, t to t + dt.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::advance(Real t, X& state)
{
    std::integral_constant<bool, DenseOutput> constexpr dense;

    auto& phi = m_scratch.phi;
    auto& kE  = m_scratch.kE;
    auto& kI  = m_scratch.kI;

    DataIterator dit = state.dataIterator();
    std::size_t const nbox = dit.size();

    #pragma omp parallel for schedule(static) 
    for (std::size_t ibox = 0; ibox < nbox; ++ibox)
    {
        // Reset the dense output coefs
        zeroDenseCoefs(ibox, dense); 

        // Set phi[0] := state, all others 0
        phi[0][ibox].copy(state[ibox]);
        for (std::size_t stage = 1; stage < s_nStages; ++stage)
            phi[stage][ibox].setVal(0.0);
    }

    advanceStage<0>(t, state);
    advanceStage<1>(t, state);
    advanceStage<2>(t, state);
    advanceStage<3>(t, state);
    advanceStage<4>(t, state);
    advanceStage<5>(t, state);
}


/*
  Advance solution state in time, t to t + dt.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <std::size_t Stage>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::advanceStage(Real t, X& state)
{
    std::integral_constant<bool, DenseOutput> constexpr dense;

    auto& phi = m_scratch.phi;
    auto& kE  = m_scratch.kE;
    auto& kI  = m_scratch.kI;

    Real stage_t = t + s_c[Stage]*m_dt;
    
    phi[Stage].exchangeSync();

    DataIterator dit = state.dataIterator();
    std::size_t const nbox = dit.size();

    #pragma omp parallel for schedule(static) 
    for (std::size_t ibox = 0; ibox < nbox; ++ibox)
    {
        auto this_state = state[ibox];

        std::array<decltype(phi[Stage][ibox]), s_nStages> this_phi;

        for (std::size_t k = 0; k < s_nStages; ++k)
            this_phi[k] = phi[k][ibox]; 

        auto this_kE = kE[ibox];
        auto this_kI = kI[ibox];

        // Calculate the operators for this stage
        m_opImEx.explicitOp<Stage>(stage_t, m_dt, s_aE, s_b
                                 , this_state, this_kE, this_phi);
        if (!m_opImEx.implicitOpDisabled())
            m_opImEx.implicitOp<Stage>(stage_t, this_kI, this_phi[Stage]);

        // Add the known stage operator values to future solves 
        for (std::size_t k = Stage + 1; k < s_nStages; ++k)
        {
            if (!m_opImEx.implicitOpDisabled())
                this_phi[k].plus(this_kI, m_dt*s_aI[k][Stage]); 
        }
    
        // Accumulate the final solution and accum registers
        if (!m_opImEx.implicitOpDisabled())
            this_state.plus(this_kI, m_dt*s_b[Stage]);
    
        accumulateDenseCoefs<Stage>(ibox, dense);

        if ((Stage + 1) < s_nStages)
        {
            Real next_t = t + s_c[Stage+1]*m_dt;

            // Do the solve - copy rhs from phi[stage]
            this_phi[Stage+1].plus(this_phi[0]);

            // Solve in-place for phi[Stage+1] in
            // (I - s_aIdiag * m_dt * FI) (phi[Stage+1]) = phi[Stage+1].
            m_opImEx.solve<Stage>(next_t, s_aIdiag, this_phi[Stage+1]);
        }
    }
}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::zeroDenseCoefs(
    std::size_t ibox
  , std::true_type
    )
{
    auto& denseCoefs = m_scratch.denseCoefs;

    for (std::size_t icoef=0; icoef < s_nDenseCoefs; ++icoef)
        denseCoefs[icoef][ibox].setVal(0.0);
}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::zeroDenseCoefs(
    std::size_t ibox
  , std::false_type
    )
{}

template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <std::size_t Stage>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::accumulateDenseCoefs(
    std::size_t ibox
  , std::true_type
    )
{
    auto& kE         = m_scratch.kE;
    auto& kI         = m_scratch.kI;
    auto& denseCoefs = m_scratch.denseCoefs;

    for (std::size_t icoef=0; icoef < s_nDenseCoefs; ++icoef)
    {
        denseCoefs[icoef][ibox].plus(kE[ibox], m_dt*s_bstar[icoef][Stage]);
        denseCoefs[icoef][ibox].plus(kI[ibox], m_dt*s_bstar[icoef][Stage]);
    }
} 

template <class X, class Scratch, class ImExOp, bool DenseOutput>
template <std::size_t Stage>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::accumulateDenseCoefs(
    std::size_t ibox
  , std::false_type
    )
{}

/*
  Return the coefs to interpolate solution, in terms of power of the fraction
  of time between t_old and t_new.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::denseOutputCoefs(
    Vector<X*>& interpCoefs
    )
{
    const std::size_t nCoef = s_nDenseCoefs+1;
    assert(interpCoefs.size() == nCoef); 

    for (std::size_t icoef = 0; icoef < nCoef; ++icoef)
        assert(interpCoefs[icoef] != NULL);

    // Copy over the dense coef values

    DataIterator dit = state.dataIterator();
    std::size_t const nbox = dit.size();

    #pragma omp parallel for schedule(static)
    for (std::size_t ibox = 0; ibox < nbox; ++ibox)
    {
        auto& phi        = m_scratch.phi;
        auto& denseCoefs = m_scratch.denseCoefs;  

        // First coefficient is just the old state
        interpCoefs[0][ibox]->copy(phi[0][ibox]);
  
        // Next coefs are our dense output
        for (int icoef = 1; icoef < nCoef ; ++icoef)
            interpCoefs[icoef][ibox]->copy(denseCoefs[icoef-1][ibox]);
    }
}

/*
  Reset the timestep.
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
void OMPARK4<X, Scratch, ImExOp, DenseOutput>::resetDt(Real dt)
{
    // Only update everything if dt has changed
    Real constexpr reltol = 1e-14;
    if (std::fabs(m_dt - dt) > m_dt*reltol)
    {
        m_dt = dt;
        m_opImEx.resetDt(m_dt);
    }
}

/*
  Static constants for OMPARK4
 */
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_aIdiag = 0.25;

// Time coefficients for each stage
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_c[] = { 0.0, 0.5, 0.332, 0.62, 0.85, 1.0 };
  
// Stage coefficients - each row is for that stage 
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_aE[][OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
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
OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_aI[][OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
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
OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_b[] =
  {0.15791629516167136, 0., 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25};

// Coefficients for dense output, 4th-order interpolation
template <class X, class Scratch, class ImExOp, bool DenseOutput>
const Real
OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_bstar[][OMPARK4<X, Scratch, ImExOp, DenseOutput>::s_nStages] = {
  {0.961753400252887, 0., 0.787405595186356, -2.74544192086633, 3.70351728061223, -1.70723435518514},
  {-1.76418754019038, 0., -0.774504669155511, 9.64023584441292, -12.544886411271, 5.44334277620397},
  {0.960350435099165, 0., 0.173858014493155, -6.21422862823726, 8.56612859966376, -3.48610842101883}
};

#include "NamespaceFooter.H"
#endif 
