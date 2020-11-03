#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

static void test (MultiFab& mfa, MultiFab const& mfb)
{
    for (MFIter mfi(mfa); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        Array4<Real const> const& b = mfb.const_array(mfi);
        Array4<Real> const& a = mfa.array(mfi);
        BL_PROFILE("daxpy"); // for NVIDIA Nsight Compute
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i,j,k) = a(i,j,k) + 0.5*b(i,j,k);
        });
    }
}

int main(int argc, char* argv[])
{
    amrex::Real t;
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
        MultiFab mfa(ba,dm,1,0);
        MultiFab mfb(ba,dm,1,0);
        mfa.setVal(1.0);
        mfb.setVal(2.0);
        {
            BL_PROFILE("daxpy-warmup");
            test(mfa, mfb);
        }
        {
            BL_PROFILE("daxpy-mf");
            amrex::Real t0 = amrex::second();
            test(mfa, mfb);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}
