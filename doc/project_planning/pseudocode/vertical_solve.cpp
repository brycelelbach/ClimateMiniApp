#if defined(CH_OPENMP)
    #define CH_CALL std::sync
    typedef std::vector<double> VColumn;
#elif defined(CH_HPX)
    #define CH_CALL hpx::dataflow
    typedef hpx::future<std::vector<double> > VColumn;
#endif

// Computes indices into 2D rhs vectors
std::size_t index(std::size_t i, std::size_t j);

// Solve the 1D problem with a 3rd-party library.
VColumn tridiagonal_solve(VColumn dl, VColumn d, VColumn du, VColumn& rhs)
{
    LAPACKE_dgtsv(/* ... */, dl.data(), d.data(), du.data(), rhs.data(), /* ... */);
    return rhs; 
}

std::vector<VColumn> implicit_solve(
    std::vector<VColumn>& rhs
    )
{
    // HPX version: Pragma is ignored
    #pragma omp parallel for schedule(dynamic)
    for (/* first horizontal dimension */)
        for (/* second horizontal dimension */)
        {
            // Build compressed tridiagonal matrix (assume this is trivial for now)
            VColumn dl = // Build subdiagonal vector (order n-1)
            VColumn d  = // Build diagonal vector (order n)
            VColumn du = // Build superdiagonal vector (order n-1)

            // HPX version: We asynchronously call tridiagonal_solve; the result
            // of the call is a future, and we store that future in rhs[idx].
            // ----
            // Serial/OMP version: We synchronously call tridiagonal_solve; the result
            // of the call is a vector of doubles, which is stored in rhs[idx].
            rhs[idx] = CH_CALL(tridiagonal_solve, dl, d, du, rhs[idx]);
        }

    // HPX version: We are returning a vector of futures; we can call when_all()
    // on this vector - so we retain composability!
    // ----
    // Serial/OMP version: We return a vector of vectors. 
    return rhs;
}

