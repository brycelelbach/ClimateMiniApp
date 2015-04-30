#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#include "MayDay.H"
#include "AMRIO.H"

#include "LapackWrapper.H"
#include "Lapack.H"

#include "NamespaceHeader.H"

void LapackWrapper::factorBandMatrix(LapackFactorization& A)
{
    // - check that the sizes of A, B are compatible
    int LDAB = A.numRows();
    int M = A.numCols(); // it's actually a square matrix
    int N = A.numCols(); // in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO;

    // Factorization
    dgbtrf_(&M, &N, &KL, &KU, A.luPtr(), &LDAB, A.pivotPtr(), &INFO);

    CH_assert(INFO == 0);
}


void LapackWrapper::solveBandMatrix(LapackFactorization& A, Real* const inout)
{
    // - check that the sizes of A, B are compatible
    int LDAB = A.numRows();
    int N = A.numCols(); // square matrix in compact band format

    int KL = A.numLower();
    int KU = A.numUpper();
    int INFO;

    // Solve using factorization
    char TRANS = 'N';
    int NRHS = 1;
    dgbtrs_(&TRANS, &N, &KL, &KU, &NRHS, A.luPtr(), 
            &LDAB, A.pivotPtr(), inout, &N, &INFO);

    CH_assert(INFO == 0);
}


void LapackWrapper::solveBandMatrix(CHMatrix& A, CHMatrix& B)
{
    // - check that the sizes of A, B are compatible
    int LDAB = A.size(0);
    int M = A.size(1);
    int N = B.size(0);
    // CH_assert(B.box().size()[0] == M); // B is mxk, A is mxn
    // CH_assert(M >= N); // A is over-determined
    // pout() << "A is " << M << " x " << N << endl;

    int KL = 1;
    int KU = 1;
    int* IPIV = new int[N];
    int INFO;

    // Factorization
    dgbtrf_(&M, &N, &KL, &KU, A.begin(), &LDAB, IPIV, &INFO);

//    pout() << "After dgbtrf:" << endl;
//    pout() << "INFO = " << INFO << endl;
    CH_assert(INFO == 0);
//    BoxIterator bit = BoxIterator(A.box());
//    for (bit.begin(); bit.ok(); ++bit)
//    {
//      IntVect iv = bit();
//      pout() << "A" << iv << " = " << A(iv) << endl;
//    }
//
//    pout() << "IPIV = (";
//    for (int i=0; i < N; ++i)
//      pout() << IPIV[i] << ((i == N-1) ? ")\n" : ", ");

    // Solve using factorization
    char TRANS = 'N';
    int NRHS = B.size(1);
    dgbtrs_(&TRANS, &N, &KL, &KU, &NRHS, A.begin(), 
            &LDAB, IPIV, B.begin(), &N, &INFO);
//    pout() << "After dgbtrs:" << endl;
//    pout() << "INFO = " << INFO << endl;
//    CH_assert(INFO == 0);
//    bit = BoxIterator(B.box());
//    for (bit.begin(); bit.ok(); ++bit)
//    {
//      IntVect iv = bit();
//      pout() << "B" << iv << " = " << B(iv) << endl;
//    }

    delete IPIV;
}


void LapackWrapper::solveBandMatrix(FArrayBox& A, FArrayBox& B)
{
    // - check that the sizes of A, B are compatible
    int LDAB = A.box().size(0);
    int M = A.box().size(0);
    int N = B.box().size(0);
    // CH_assert(B.box().size()[0] == M); // B is mxk, A is mxn
    // CH_assert(M >= N); // A is over-determined
    // pout() << "A is " << M << " x " << N << endl;

    int KL = 1;
    int KU = 1;
    int* IPIV = new int[N];
    int INFO;

    // Factorization
    dgbtrf_(&M, &N, &KL, &KU, A.dataPtr(), &LDAB, IPIV, &INFO);

//    pout() << "After dgbtrf:" << endl;
//    pout() << "INFO = " << INFO << endl;
    CH_assert(INFO == 0);
//    BoxIterator bit = BoxIterator(A.box());
//    for (bit.begin(); bit.ok(); ++bit)
//    {
//      IntVect iv = bit();
//      pout() << "A" << iv << " = " << A(iv) << endl;
//    }
//
//    pout() << "IPIV = (";
//    for (int i=0; i < N; ++i)
//      pout() << IPIV[i] << ((i == N-1) ? ")\n" : ", ");

    // Solve using factorization
    char TRANS = 'N';
    int NRHS = B.box().size(1);
    dgbtrs_(&TRANS, &N, &KL, &KU, &NRHS, A.dataPtr(), 
            &LDAB, IPIV, B.dataPtr(), &N, &INFO);
//    pout() << "After dgbtrs:" << endl;
//    pout() << "INFO = " << INFO << endl;
//    CH_assert(INFO == 0);
//    bit = BoxIterator(B.box());
//    for (bit.begin(); bit.ok(); ++bit)
//    {
//      IntVect iv = bit();
//      pout() << "B" << iv << " = " << B(iv) << endl;
//    }

    delete IPIV;
}

#include "NamespaceFooter.H"
