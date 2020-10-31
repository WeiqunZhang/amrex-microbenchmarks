#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

static void test (MultiFab const& rhs, MultiFab& mfa)
{
    for (MFIter mfi(mfa); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const Box& gbx = mfi.fabbox();
        Array4<Real const> const& r = rhs.const_array(mfi);
        Array4<Real> const& a = mfa.array(mfi);
        FArrayBox bfab(gbx, 1);
        Array4<Real> const& b = bfab.array();
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            b(i,j,k) = a(i,j,k);
        });
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i,j,k) += (2./3.) * r(i,j,k) /
                (b(i-1,j,k)+b(i+1,j,k)+b(i,j-1,k)+b(i,j+1,k)+b(i,j,k-1)+b(i,j,k+1)-6.*b(i,j,k));
        });
        Gpu::synchronize();
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int max_grid_size = 256;
        {
            ParmParse pp;
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(255));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};
        MultiFab rhs(ba,dm,1,0);
        MultiFab mfa(ba,dm,1,1);
        rhs.setVal(1.3);
        mfa.setVal(1.0);
        {
            BL_PROFILE("jacobi-warmup");
            test(rhs, mfa);
        }
        {
            BL_PROFILE("jacobi-mf");
            test(rhs, mfa);
        }
    }
    amrex::Finalize();
}
