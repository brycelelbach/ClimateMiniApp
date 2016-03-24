#if !defined(CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D)
#define CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D

#include <array>

#include "ARK4.H"

#include "CMAProblemState.H"

namespace climate_mini_app
{

template <bool DenseOutput>
struct problem_state_scratch;

template <>
struct problem_state_scratch<true> 
{
    std::array<problem_state, ARK4Sizes::s_nStages>
        phi;
    std::array<problem_state, ARK4Sizes::s_nDenseCoefs>
        denseCoefs;

    problem_state kE;
    problem_state kI;

    void define(
        DisjointBoxLayout const& dbl
      , int comps
      , IntVect ghost
      , std::uint64_t tile_width
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i].define(dbl, comps, ghost, tile_width); 

        for (std::size_t i = 0; i < denseCoefs.size(); ++i)
            denseCoefs[i].define(dbl, comps, ghost, tile_width);

        kE.define(dbl, comps, ghost, tile_width);
        kI.define(dbl, comps, ghost, tile_width);
    }
};

template <>
struct problem_state_scratch<false> 
{
    std::array<problem_state, ARK4Sizes::s_nStages>
        phi;

    problem_state kE;
    problem_state kI;

    void define(
        DisjointBoxLayout const& dbl
      , int comps
      , IntVect ghost
      , std::uint64_t tile_width
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i].define(dbl, comps, ghost, tile_width); 

        kE.define(dbl, comps, ghost, tile_width);
        kI.define(dbl, comps, ghost, tile_width);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <bool DenseOutput>
struct problem_state_fab_scratch;

template <>
struct problem_state_fab_scratch<true> 
{
    std::array<problem_state_fab, ARK4Sizes::s_nStages>
        phi;
    std::array<problem_state_fab, ARK4Sizes::s_nDenseCoefs>
        denseCoefs;

    problem_state_fab kE;
    problem_state_fab kI;

    void define(
        DataIndex di
      , std::array<problem_state, ARK4Sizes::s_nStages>& phi_ 
      , std::array<problem_state, ARK4Sizes::s_nDenseCoefs>& denseCoefs_
      , problem_state& kE_
      , problem_state& kI_
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] = problem_state_fab(phi_[i], di);

        for (std::size_t i = 0; i < denseCoefs.size(); ++i)
            denseCoefs[i] = problem_state_fab(denseCoefs_[i], di);

        kE = problem_state_fab(kE_, di);
        kI = problem_state_fab(kI_, di);
    }
};

template <>
struct problem_state_fab_scratch<false> 
{
    std::array<problem_state_fab, ARK4Sizes::s_nStages>
        phi;

    problem_state_fab kE;
    problem_state_fab kI;

    void define(
        DataIndex di
      , std::array<problem_state, ARK4Sizes::s_nStages>& phi_ 
      , problem_state& kE_
      , problem_state& kI_
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] = problem_state_fab(phi_[i], di);

        kE = problem_state_fab(kE_, di);
        kI = problem_state_fab(kI_, di);
    }
};

}

#endif // CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D

