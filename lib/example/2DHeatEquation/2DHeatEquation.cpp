#include <boost/format.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#include <lapacke.h>

#include "FArrayBox.H"

Real constexpr nt = 0.003; 
size_t constexpr ns = 100;

size_t constexpr nx = 20;
size_t constexpr ny = 20; 

Real constexpr kx = 0.5;
Real constexpr ky = 0.75;

Real constexpr dt = nt/ns; // TODO: actually compute CFL
Real constexpr dx = 1.0/(1.0*nx-1.0);
Real constexpr dy = 1.0/(1.0*ny-1.0);

Real initial_profile(Real t, size_t i, size_t j)
{
    return 0.0;
}

void verify(bool predicate, std::string const& fail_msg)
{
    if (!predicate)
        throw std::runtime_error(fail_msg);
}

std::tuple<Real, Real> phys_coords(size_t i, size_t j)
{
    return std::tuple<Real, Real>(Real(i)*dx, Real(j)*dy);
}

///////////////////////////////////////////////////////////////////////////////
Real explicit_op(Real c, Real here, Real left, Real right)
{
    return here + c*(left - 2*here + right); 
}

Real source(size_t i, size_t j)
{
    Real x, y;
    std::tie(x, y) = phys_coords(i, j);
    return std::sin(M_PI*x)*std::sin(M_PI*y);
} 

Real analytic(Real t, size_t i, size_t j) 
{
    Real constexpr pi_sq = M_PI*M_PI;
    Real const a = (1.0 - std::exp(-(kx+ky)*pi_sq*t)) / ((kx+ky)*pi_sq);
    return a*source(i, j);
} 

///////////////////////////////////////////////////////////////////////////////
// Boundary conditions.

Real left_boundary(Real c, Real here, Real right)
{
    return here; 
}

Real right_boundary(Real c, Real here, Real left)
{
    return here;
}

// Returns [d.front(), du.front()]
std::tuple<Real, Real> down_boundary(Real c, Real here, Real up)
{
    // Should be [1.0+c, -c/2] for a boundary condition of 0.
    return std::tuple<Real, Real>{1.0+c, -c/2};
}

// Returns [d.back(), dl.back()]
std::tuple<Real, Real> up_boundary(Real c, Real here, Real down)
{
    // Should be [1.0+c, -c/2] for a boundary condition of 0.
    return std::tuple<Real, Real>{1.0+c, -c/2}; 
}

void advance(FArrayBox& soln)
{
    IntVect lower = soln.smallEnd();
    IntVect upper = soln.bigEnd(); 
    Interval comps = soln.interval();

    // Dimensions of the problem.
    size_t const nx = upper[0]-lower[0]; 
    size_t const ny = upper[1]-lower[1]; 

    Real const cx = kx*dt/(dx*dx);
    Real const cy = ky/(dy*dy);

    // Storage for the RHS. 
    std::vector<std::vector<Real> > RHS(
        nx, // Number of "columns" to solve
        std::vector<Real>(ny, 0.0) // Two extra equations (boundary conditions) 
    );

    // Explicit op.
    for (size_t j = lower[1]; j < upper[1]; ++j)
    {
        // Adjusted indices.
        size_t jj = j - lower[1];

        ///////////////////////////////////////////////////////////////////////
        // Left/right boundary conditions
        IntVect leftmost   (lower[0]  , j  , 0);
        IntVect lm_neighbor(lower[0]+1, j  , 0);
        IntVect rightmost  (upper[0]-1, j  , 0);
        IntVect rm_neighbor(upper[0]-2, j  , 0);
 
        RHS[lower[0]  ][jj] = left_boundary (cx, soln(leftmost),  soln(lm_neighbor));
        RHS[upper[0]-1][jj] = right_boundary(cx, soln(rightmost), soln(rm_neighbor));

        ///////////////////////////////////////////////////////////////////////
        // Interior points.
        for (size_t i = lower[0]+1; i < upper[0]-1; ++i)
        {
            // Adjusted indices for RHS.
            size_t ii = i - lower[0];

            IntVect here (i  , j  , 0);
            IntVect left (i-1, j  , 0);
            IntVect right(i+1, j  , 0);

            RHS[ii][jj] = explicit_op(cx, soln(here), soln(left), soln(right))
                        + dt*source(i, j); 
        }
    }

    // Sum into solution (for storage).
    for (size_t j = lower[1]; j < upper[1]; ++j)
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            // Adjusted indices.
            size_t ii = i - lower[0];
            size_t jj = j - lower[1];

            IntVect here(i, j, 0);
 
            soln(here) = RHS[ii][jj]; 
        }

    // Now we're going to solve a bunch of 1D diffusion problems in the 
    // vertical direction.

    // For each column, we want to solve Ax=RHS, where A is a matrix of the
    // form: 
    //
    //  1+c  -c/2   0  
    // -c/2   1+c -c/2
    //   0   -c/2  1+c

    // Implicit op.
    for (size_t i = lower[0]; i < upper[0]; ++i)
    {
        // Sub-diagonal part of the matrix.
        std::vector<Real> dl(ny-1, -cy/2.0);
        // Diagonal part of the matrix.
        std::vector<Real> d(ny, 1.0+cy);
        // Super-diagonal part of the matrix.
        std::vector<Real> du(ny-1, -cy/2.0);

        // Adjusted indices.
        size_t ii = i - lower[0]; 

        ///////////////////////////////////////////////////////////////////////
        // Up/down boundary conditions
         
        size_t downmost   (0               );
        size_t dm_neighbor(0+1             );
        size_t upmost     (RHS[ii].size()-1);
        size_t um_neighbor(RHS[ii].size()-2);
 
//        std::tie(d.front(), du.front()) = down_boundary(cy, RHS[ii][downmost], RHS[ii][dm_neighbor]);
//        std::tie(d.back(), dl.back())   = up_boundary  (cy, RHS[ii][upmost],   RHS[ii][um_neighbor]);

        du.front() = 0.0; d.front() = 0.0; dl.front() = 0.0;
        du.back() = 0.0; d.back() = 0.0; dl.back() = 0.0;

        ///////////////////////////////////////////////////////////////////////
        // Solve each tridiagonal system.
        LAPACKE_dgtsv(
            LAPACK_ROW_MAJOR, // matrix format
            ny, // matrix order
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
            size_t jj = j - lower[1];

            IntVect here(i, j, 0);

            soln(here) = 0.5*(soln(here) + RHS[ii][jj]); 
        }
}

template <typename F>
void init_matrix(FArrayBox& soln, Real t, F f)
{
    IntVect lower = soln.smallEnd();
    IntVect upper = soln.bigEnd(); 

    for (size_t j = lower[1]; j < upper[1]; ++j)
    {
        for (size_t i = lower[0]; i < upper[0]; ++i)
        {
            IntVect here(i, j, 0);

            soln(here) = f(t, i, j);
        }
    }
}

void print_matrix(FArrayBox const& soln, size_t timestep, std::string const& file_template)
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
            ofs << " " << i << " " << j << " " << soln(here) << "\n";
        }

        ofs << "\n";
    }
}

int main()
{
    Box b(IntVect(0,0,0), IntVect(nx,ny,1));

    FArrayBox soln(b, 1);
    init_matrix(soln, 0.0, initial_profile);
    print_matrix(soln, 0, "numeric.%04i.dat");

    FArrayBox exact(b, 1);

    init_matrix(exact, 0.0, analytic);
    print_matrix(exact, 0, "analytic.%04i.dat"); 

    exact -= soln; 
    print_matrix(exact, 0, "error.%04i.dat"); 

    for (size_t t = 1; t < ns + 1; ++t)
    {
        advance(soln);
        print_matrix(soln, t, "numeric.%04i.dat");

        init_matrix(exact, t * dt, analytic);
        print_matrix(exact, t, "analytic.%04i.dat"); 

        exact -= soln; 
        print_matrix(exact, t, "error.%04i.dat"); 
    }
}
