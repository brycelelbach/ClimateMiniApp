#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _LAPACKFACTORIZATION_H_
#define _LAPACKFACTORIZATION_H_

#include <assert.h>

#include "NamespaceHeader.H"

// Class to cache a factorization from Lapack
class LapackFactorization
{
  public:
    // Set reasonable defaults
    LapackFactorization() 
    {
        m_ncols = 0;
        m_kl = 0;
        m_ku = 0;
        m_nrows = 0;
        m_LU = NULL;
        m_pivot = NULL;
    }

    LapackFactorization(int a_ncols, int a_kl, int a_ku)
    {
        m_LU = NULL;
        m_pivot = NULL;
        define(a_ncols, a_kl, a_ku);
    }
   
    ~LapackFactorization() 
    {
        if (m_LU) delete [] m_LU;
        if (m_pivot) delete [] m_pivot;
    }

    // Allocate memory
    void define(int a_ncols, int a_kl, int a_ku)
    {
        assert(a_ku >= 0);
        assert(a_kl >= 0);
        assert(a_ncols > 0);
        m_ncols = a_ncols;
        m_kl = a_kl;
        m_ku = a_ku;
        m_nrows = 2*m_kl + m_ku + 1; // Band storage

        if (m_LU) delete [] m_LU;
        if (m_pivot) delete [] m_pivot;

        m_LU = new Real[m_nrows*m_ncols];
        m_pivot = new int[m_ncols];

        setZero();
    }

    void setZero()
    {
        for(int i=0;i<m_nrows*m_ncols;i++)
            m_LU[i]=0;
    }

    int numUpper()
    {
        return m_ku;
    }

    int numLower()
    {
        return m_kl;
    }

    int numRows()
    {
        return m_nrows;
    }

    int numCols()
    {
        return m_ncols;
    }

    // Assignment operator, copies data
    void define(const LapackFactorization& a_in)
    {
        // Resize to match the input 
        define(a_in.m_ncols, a_in.m_kl, a_in.m_ku);
        setZero();

        // Copy the data
        for (int ix=0; ix < m_nrows*m_ncols; ix++)
            m_LU[ix] = a_in.m_LU[ix];
        for (int ix=0; ix < m_ncols; ix++)
            m_pivot[ix] = a_in.m_pivot[ix];
    }

    // Getting pointer access for Lapack factorization matrix
    int* pivotPtr() const
    {
      return m_pivot;
    }

    // Getting pointer access for Lapack pivot array
    Real* luPtr() const
    {
      return m_LU;
    }

    void printBandedMatrix()
    {
        //pout() << std::setprecision(16);
        //pout() << "banded matrix " << m_ncols << "x" << m_ncols << " " << m_kl << " " << m_ku << " " << m_nrows << endl;
        for (int i=0;i<m_ncols;i++)
        {
            for (int j=0;j<m_ncols;j++)
            {
                int band=j-i;
                //pout() << "band " << band << endl;

                if (band>=-m_kl && band<=m_ku)
                {
                    //int abrow = m_kl+m_ku+1-band;
                    //pout() << m_LU[abrow-1+j*m_nrows];
                    std::cout << (*this)(i,j);
                }

                else
                    std::cout << 0;

                if (j==(m_ncols-1))
                    std::cout << ";";

                else std::cout << ",";
            }

            std::cout << endl;
        }
    }
  
    Real& operator()(const int i,const int j)
    {
        assert((i-j)<=m_kl);
        assert((j-i)<=m_ku);
        assert(i>=0);
        assert(i<m_ncols);
        assert(j>=0);
        assert(j<m_ncols);
        return m_LU[m_kl+m_ku-j+i+j*m_nrows];
    }

    Real get(const int i,const int j)
    {
        return (*this)(i,j);
    }

    void multiply(Real y[],const Real x[])
    {
        for (int i=0;i<m_ncols;i++)
        {
            Real yv=0;
            for (int j=Max(0,i-m_kl);j<=Min(m_ncols-1,i+m_ku);j++)
                yv+=get(i,j)*x[j];
            y[i]=yv;
        }
    }

    bool is_cached()
    {
        return m_LU && m_pivot;
    }

  protected:
    LapackFactorization(const LapackFactorization&);
    void operator=(const LapackFactorization&);

  private:
    int m_ncols;
    int m_ku;
    int m_kl;
    int m_nrows;
    Real* m_LU;
    int* m_pivot;
};

#include "NamespaceFooter.H"

#endif
