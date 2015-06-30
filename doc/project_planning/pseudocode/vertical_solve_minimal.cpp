#if defined(CH_OPENMP)
    #define CH_CALL std::sync
    typedef std::vector<double> VColumn;
#elif defined(CH_HPX)
    #define CH_CALL hpx::dataflow
    typedef hpx::future<std::vector<double> > VColumn;
#endif

extern std::size_t index(std::size_t i, std::size_t j);

extern VColumn tridiagonal_solve(VColumn dl, VColumn d, VColumn du, VColumn& rhs);

std::vector<VColumn> implicit_solve(std::vector<VColumn>& rhs)
{
    #pragma omp parallel for schedule(dynamic) // Ignored in HPX build
    for (/* first horizontal dimension */)
        for (/* second horizontal dimension */)
        {
            VColumn dl = // Build subdiagonal vector (order n-1)
            VColumn d  = // Build diagonal vector (order n)
            VColumn du = // Build superdiagonal vector (order n-1)

            rhs[idx] = CH_CALL(tridiagonal_solve, dl, d, du, rhs[idx]);
        }

    return rhs;
}
