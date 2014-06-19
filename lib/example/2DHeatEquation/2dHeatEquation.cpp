#include <boost/format.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#include <lapacke.h>

#include "FArrayBox.H"

Real constexpr T_floor = 1e-10; // Temperature floor.
  
Real constexpr nt = 5; 
Real constexpr nx = 100;
Real constexpr ny = 10; 

Real constexpr k = 0.5; // Diffusion coefficient 
Real constexpr A = 0.25; // Parameter of the initial state 

Real initial_profile(Real nx, Real x, Real ny, Real y)
{
    Real a = 500.0/(nx*nx);
    Real c = 20.0/(ny*ny);
    Real x0 = (nx-1.0)/2.0;
    Real y0 = (ny-1.0)/2.0;
    Real T = A*std::exp(a*(x-x0)-c*(std::fabs(y-y0)));
    if (T < T_floor) return T_floor; else return T; 
}

Real const max_temp(initial_profile(nx, nx-1, ny, ny-1));

void verify(bool predicate, std::string const& fail_msg)
{
    if (!predicate)
        throw std::runtime_error(fail_msg);
}

///////////////////////////////////////////////////////////////////////////////
Real explicit_op(Real z, Real here, Real left, Real right)
{
    return here + z*(left - 2*here + right); 
}

///////////////////////////////////////////////////////////////////////////////
// Boundary conditions.

// The domain is a tube which is open on one end (the left). The up/down
// BCs are reflecting; the right BC  

Real left_boundary(Real z, Real here, Real right)
{
    // Assume the temperature outside of the tube is 0. 
    return here + z*(-2*here + right); 
}

Real right_boundary(Real z, Real here, Real left)
{
    // Heat source to the right. 
    return here + 0.5*z*(left - 2*here + max_temp); 
}

void stencil(FArrayBox& soln)
{
    IntVect lower = soln.smallEnd();
    IntVect upper = soln.bigEnd(); 
    Interval comps = soln.interval();

    // Dimensions of the problem.
    size_t const nx = upper[0]-lower[0]; 
    size_t const ny = upper[1]-lower[1]; 

    Real constexpr dt = 1.0; // TODO: actually compute CFL
    Real constexpr dh = 1.0;

    Real const z = k*dt/(dh*dh);

    // Storage for the RHS. 
    std::vector<std::vector<Real> > RHS(
        nx, // Number of "columns" to solve
        std::vector<Real>(ny+2, 0.0) // Two extra equations (boundary conditions) 
    );

    // Explicit op.
    for (size_t j = lower[1]; j < upper[1]; ++j)
    {
        // Adjusted indices.
        size_t jj = j - lower[1] + 1;

        ///////////////////////////////////////////////////////////////////////
        // Left/right boundary conditions
        IntVect leftmost   (lower[0]  , j  , 0);
        IntVect lm_neighbor(lower[0]+1, j  , 0);
        IntVect rightmost  (upper[0]-1, j  , 0);
        IntVect rm_neighbor(upper[0]-2, j  , 0);
 
        RHS[lower[0]  ][jj] = left_boundary (z, soln(leftmost), soln(lm_neighbor));
        RHS[upper[0]-1][jj] = right_boundary(z, soln(rightmost), soln(rm_neighbor));

        ///////////////////////////////////////////////////////////////////////
        // Interior points.
        for (size_t i = lower[0]+1; i < upper[0]-1; ++i)
        {
            // Adjusted indices for RHS.
            size_t ii = i - lower[0];

            IntVect here (i  , j  , 0);
            IntVect left (i-1, j  , 0);
            IntVect right(i+1, j  , 0);

            RHS[ii][jj] = explicit_op(z, soln(here), soln(left), soln(right)); 
        }
    }

    // Sum into solution (for storage).
    for (size_t j = lower[1]; j < upper[1]; ++j)
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            // Adjusted indices.
            size_t ii = i - lower[0];
            size_t jj = j - lower[1] + 1;

            IntVect here(i, j, 0);
 
            soln(here) = RHS[ii][jj]; 
        }

    // Now we're going to solve a bunch of 1D diffusion problems in the 
    // vertical direction.

    // For each column, we want to solve Ax=RHS, where A is a matrix of the
    // form: 
    //
    //   1   -z/2   0  
    // -z/2    1  -z/2
    //   0   -z/2   1

    // Implicit op.
    for (size_t i = lower[0]; i < upper[0]; ++i)
    {
        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(ny+1, -z/2.0);
        // Diagonal part of the matrix.
        std::vector<Real> d(ny+2, 1.0+z);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(ny+1, -z/2.0);

        // Adjusted indices.
        size_t ii = i - lower[0]; 

        ///////////////////////////////////////////////////////////////////////
        // Up/down boundary conditions
         
        // First, we need to adjust the matrix for the boundary conditions.

        // The "down" boundary is the first equation. I /think/ it should look
        // like [ 1 -z 0 ] for reflecting BCs.
        du.front()      = -z;
        RHS[ii].front() = RHS[ii][lower[1]+1];

        // Up boundary should be [ 0 -z 1 ]
        dl.back()       = -z;
        RHS[ii].back()  = RHS[ii][upper[1]-2];

        // Solve each tridiagonal system.
        LAPACKE_dgtsv(
            LAPACK_ROW_MAJOR, // matrix format
            ny+2, // matrix order
            1, // # of right hand sides 
            dl.data(), // subdiagonal part
            d.data(), // diagonal part
            du.data(), // superdiagonal part
            RHS[ii].data(), // column to solve 
            1 // leading dimension of RHS
            );
    }

    // Finalize this step. 
    for (size_t j = lower[1]; j < upper[1]; ++j)
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            // Adjusted indices.
            size_t ii = i - lower[0];
            size_t jj = j - lower[1] + 1;

            IntVect here(i, j, 0);

            soln(here) = 0.5*(soln(here) + RHS[ii][jj]); 
        }
}

void init_matrix(FArrayBox& soln)
{
    IntVect lower = soln.smallEnd();
    IntVect upper = soln.bigEnd(); 

    for (size_t j = lower[1]; j < upper[1]; ++j)
    {
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            IntVect here(i, j, 0);

            // Tube, open on the left, closed on the other sides.
            soln(here) = initial_profile(nx, i, ny, j);

/*
            // Gaussian blob. 
            Real constexpr A = 1.5;
            Real const a = 100.0/(nx*nx);
            Real const c = 100.0/(ny*ny);

            Real x0 = (nx-1.0)/2.0;
            Real y0 = (nx-1.0)/2.0;

            soln(here) = A*std::exp(-(a*(i-x0)*(i-x0)+c*(j-y0)*(j-y0)));

            if (soln(here) < floor)
                soln(here) = floor;

            // Boundaries should be zero.
            if (i==0||j==0||(i==ny-1)||(j==ny-1))
                verify(std::fabs(soln(here)-floor) < 1e-16,
                       "Initial profile yielded non-zero boundaries."); 
*/
        }
    }
}

void print_matrix(FArrayBox const& soln, size_t timestep, std::string const& file_template = "U.%04i.dat")
{
    std::string file = boost::str(boost::format(file_template) % timestep);
    std::ofstream ofs(file);

    IntVect lower = soln.smallEnd();
    IntVect upper = soln.bigEnd(); 

    for (size_t j = lower[1]; j < upper[1]; ++j)
    {
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            IntVect here(i, j, 0);
            ofs << i << " " << j << " " << soln(here) << "\n";
        }

        ofs << "\n";
    }
}

int main()
{
    Box b(IntVect(0,0,0), IntVect(nx,ny,1));

    FArrayBox soln(b, 1);
    init_matrix(soln);
    print_matrix(soln, 0);

    for (size_t t = 0; t < nt; ++t)
    {
        stencil(soln);
        print_matrix(soln, t + 1);
    }
}
