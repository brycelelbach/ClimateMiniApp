#if !defined(CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D)
#define CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D

#include <array>

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
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i].define(dbl, comps, ghost); 

        for (std::size_t i = 0; i < denseCoefs.size(); ++i)
            denseCoefs[i].define(dbl, comps, ghost);

        kE.define(dbl, comps, ghost);
        kI.define(dbl, comps, ghost);
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
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i].define(dbl, comps, ghost); 

        kE.define(dbl, comps, ghost);
        kI.define(dbl, comps, ghost);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <bool DenseOutput>
struct sub_problem_state_scratch;

template <>
struct sub_problem_state_scratch<true> 
{
    std::array<sub_problem_state, ARK4Sizes::s_nStages>
        phi;
    std::array<sub_problem_state, ARK4Sizes::s_nDenseCoefs>
        denseCoefs;

    sub_problem_state kE;
    sub_problem_state kI;

    void define(
        DataIndex di
      , std::array<problem_state, ARK4Sizes::s_nStages>& phi_ 
      , std::array<problem_state, ARK4Sizes::s_nDenseCoefs>& denseCoefs_
      , problem_state& kE_
      , problem_state& kI_
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] = sub_problem_state(phi_[i], di);

        for (std::size_t i = 0; i < denseCoefs.size(); ++i)
            denseCoefs[i] = sub_problem_state(denseCoefs_[i], di);

        kE = sub_problem_state(kE_, di);
        kI = sub_problem_state(kI_, di);
    }
};

template <>
struct sub_problem_state_scratch<false> 
{
    std::array<sub_problem_state, ARK4Sizes::s_nStages>
        phi;

    sub_problem_state kE;
    sub_problem_state kI;

    void define(
        DataIndex di
      , std::array<problem_state, ARK4Sizes::s_nStages>& phi_ 
      , problem_state& kE_
      , problem_state& kI_
        )
    {  
        for (std::size_t i = 0; i < phi.size(); ++i)
            phi[i] = sub_problem_state(phi_[i], di);

        kE = sub_problem_state(kE_, di);
        kI = sub_problem_state(kI_, di);
    }
};

}

#endif // CHOMBO_19D5911F_78B9_4437_B0D5_1A11DB59724D

