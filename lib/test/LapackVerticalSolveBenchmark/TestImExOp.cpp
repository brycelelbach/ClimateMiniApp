#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "Lapack.H"
#include "RealVect.H"
#include "BoxIterator.H"
#include "TestImExOp.H"

#include "NamespaceHeader.H"

void TestImExOp::exact(TestData& a_exact, Real a_time) const
{
    LevelData<FArrayBox>& exactLD = a_exact.U;

    Box const domain = a_exact.domain();
    RealVect const dx = 1.0 / RealVect(domain.size());

    // Calculate the amplification factor for this wave number
    Real const gamma = cI * 2.0 * (std::cos(kz * dx[0]) - 1.0) / (dx[0] * dx[0]);

    // How many steps have we taken?
    Real const steps = a_time / m_dt;

    // Calculate the cos() exact solution scaling for this many steps, e.g.
    // (1 / (1 - gamma * dt)) ^ steps
    Real const exactScale = std::pow(1.0 - gamma * m_dt, -steps); 

    DataIterator dit = exactLD.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox& exactFAB = exactLD[dit];
        Box b = exactLD.disjointBoxLayout()[dit];
        for (BoxIterator bit(b); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect xyz = (RealVect(iv) + 0.5) * dx;
            exactFAB(iv, 0) = exactScale
                            * std::cos(kx * xyz[2])
                            * std::cos(ky * xyz[1])
                            * std::cos(kz * xyz[0]);
        }
    }
}

void TestImExOp::implicitSolve(
    std::pair<DataIndex, Box> const& a_tile
  , TestData& a_state
  , Real a_time
    ) const
{
    Box const domain = a_state.domain(); 
    RealVect const dx = 1.0 / RealVect(domain.size());

    // These are the constants for a 2nd-order (I - dt*cI*Dzz) operator.
    Real const c1 = 1;
    Real const c2 = -cI * m_dt;

    DataIndex dataix = a_tile.first;
    Box b = a_tile.second;
    FArrayBox& A = a_state.A[dataix];
    FArrayBox& B = a_state.B[dataix];
    FArrayBox& C = a_state.C[dataix];
    FArrayBox& U = a_state.U[dataix];

    buildTridiagonal(A, B, C, b, dx[0], c1, c2);
    implicitSolveTridiag(A, B, C, U, b);
}

void TestImExOp::buildTridiagonal(
    FArrayBox& a_A
  , FArrayBox& a_B
  , FArrayBox& a_C
  , Box a_box
  , Real a_dx
  , Real a_c1
  , Real a_c2
    ) const
{
    Real const coef = a_c2 / (1.0 * a_dx * a_dx);

    IntVect const lo = a_box.smallEnd();
    IntVect const hi = a_box.bigEnd();

    for (int i = lo[2]; i <= hi[2]; ++i)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* ap = &a_A(IntVect(lo[0], j, i));
            Real* bp = &a_B(IntVect(lo[0], j, i));
            Real* cp = &a_C(IntVect(lo[0], j, i));

            __assume_aligned(ap, 64);
            __assume_aligned(bp, 64);
            __assume_aligned(cp, 64);

            #pragma simd
            for (int k = 0; k <= hi[0] - lo[0]; ++k)
            {
                bp[k] = a_c1 - 2.0 * coef;
            }

            #pragma simd
            for (int k = 0; k <= hi[0] - lo[0] - 1; ++k)
            {
                ap[k] = 1.0 * coef;
                cp[k] = 1.0 * coef;
            }
        }

    // Fix the diagonal ends for the homogeneous Neumann BC's
    for (int i = lo[2]; i <= hi[2]; ++i)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            a_B(IntVect(lo[0], j, i)) = a_c1 - 1.0 * coef;
            a_B(IntVect(hi[0], j, i)) = a_c1 - 1.0 * coef;
        }
}

void TestImExOp::implicitSolveTridiag(
    FArrayBox& a_A
  , FArrayBox& a_B 
  , FArrayBox& a_C 
  , FArrayBox& a_U 
  , Box a_box
    ) const
{
    int nz = a_box.size(0); 

    IntVect const lo = a_box.smallEnd();
    IntVect const hi = a_box.bigEnd();

    for (int i = lo[2]; i <= hi[2]; ++i)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            int info = 0;
            int nrhs = 1;
            int ldb = nz;

            dgtsv_(
                &nz // matrix order
              , &nrhs // # of right hand sides 
              , &a_A(IntVect(lo[0], j, i)) // subdiagonal part
              , &a_B(IntVect(lo[0], j, i)) // diagonal part
              , &a_C(IntVect(lo[0], j, i)) // superdiagonal part
              , &a_U(IntVect(lo[0], j, i)) // column to solve 
              , &ldb // leading dimension of RHS
              , &info
            );

            assert(info == 0);
        }
}

#include "NamespaceFooter.H"

