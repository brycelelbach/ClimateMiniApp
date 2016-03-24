#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

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
    Real const gamma = cI * 2.0 * (std::cos(kz * dx[2]) - 1.0) / (dx[2] * dx[2]);

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
                            * std::cos(kx * xyz[0])
                            * std::cos(ky * xyz[1])
                            * std::cos(kz * xyz[2]);
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

    buildTridiagonal(A, B, C, b, dx[2], c1, c2);
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

    for (int k = lo[2]; k <= hi[2]; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* bp = &a_B(IntVect(lo[0], j, k));

            __assume_aligned(bp, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
                bp[i] = a_c1 - 2.0 * coef;
            }
        }

    for (int k = lo[2]; k <= hi[2] - 1; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* ap = &a_A(IntVect(lo[0], j, k));
            Real* cp = &a_C(IntVect(lo[0], j, k));

            __assume_aligned(ap, 64);
            __assume_aligned(cp, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
                ap[i] = 1.0 * coef;
                cp[i] = 1.0 * coef;
            }
        }

    // Fix the diagonal ends for the homogeneous Neumann BC's
    for (int j = lo[1]; j <= hi[1]; ++j)
    {
        Real* bbeginp = &a_B(IntVect(lo[0], j, lo[2]));
        Real* bendp   = &a_B(IntVect(lo[0], j, hi[2]));

        __assume_aligned(bbeginp, 64);
        __assume_aligned(bendp, 64);

        #pragma simd
        for (int i = 0; i <= hi[0] - lo[0]; ++i)
        {
            bbeginp[i] = a_c1 - 1.0 * coef;
            bendp[i]   = a_c1 - 1.0 * coef;
        }
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
    IntVect const lo = a_box.smallEnd();
    IntVect const hi = a_box.bigEnd();

    // Forward elimination. 
    for (int k = lo[2] + 1; k <= hi[2]; ++k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* up     = &a_U(IntVect(lo[0], j, k));
            Real* usub1p = &a_U(IntVect(lo[0], j, k - 1));
            Real* asub1p = &a_A(IntVect(lo[0], j, k - 1));
            Real* bp     = &a_B(IntVect(lo[0], j, k));
            Real* bsub1p = &a_B(IntVect(lo[0], j, k - 1));
            Real* csub1p = &a_C(IntVect(lo[0], j, k - 1));

            __assume_aligned(up, 64);
            __assume_aligned(usub1p, 64);
            __assume_aligned(asub1p, 64);
            __assume_aligned(bp, 64);
            __assume_aligned(bsub1p, 64);
            __assume_aligned(csub1p, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
                // double const m = a[k - 1] / b[k - 1];
                Real const m = asub1p[i] / bsub1p[i];
                // b[k] -= m * c[k - 1];
                bp[i] -= m * csub1p[i];
                // u[k] -= m * u[k - 1];
                up[i] -= m * usub1p[i];
            }
        }

    for (int j = lo[1]; j <= hi[1]; ++j)
    {
        Real* uendp = &a_U(IntVect(lo[0], j, hi[2]));
        Real* bendp = &a_B(IntVect(lo[0], j, hi[2]));

        __assume_aligned(uendp, 64);
        __assume_aligned(bendp, 64);

        for (int i = 0; i <= hi[0] - lo[0]; ++i)
        {
            // u[nz - 1] = u[nz - 1] / b[nz - 1];
            uendp[i] = uendp[i] / bendp[i];
        }
    }
 
    // Back substitution. 
    for (int k = hi[2] - 1; k >= lo[2]; --k)
        for (int j = lo[1]; j <= hi[1]; ++j)
        {
            Real* up     = &a_U(IntVect(lo[0], j, k));
            Real* uadd1p = &a_U(IntVect(lo[0], j, k + 1));
            Real* bp     = &a_B(IntVect(lo[0], j, k));
            Real* cp     = &a_C(IntVect(lo[0], j, k));

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);
            __assume_aligned(bp, 64);
            __assume_aligned(cp, 64);

            #pragma simd
            for (int i = 0; i <= hi[0] - lo[0]; ++i)
            {
                // u[k] = (u[k] - c[k] * u[k + 1]) / b[k];
                up[i] = (up[i] - cp[i] * uadd1p[i]) / bp[i];
            }
        }
}

#include "NamespaceFooter.H"

