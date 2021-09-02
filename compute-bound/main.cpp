#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

using namespace amrex;

static void test (MultiFab& mf)
{
    if (mf.isFusingCandidate()) {
        auto const& ma = mf.arrays();
        amrex::ParallelFor(mf,
        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) noexcept
        {
            Real y = ma[b](i,j,k);
            Real x = 1.0;
            for (int n = 0; n < 20; ++n) {
                Real dx = -(x*x-y) / (2.*x);
                x += dx;
            }
            ma[b](i,j,k) = x;
        });
        Gpu::streamSynchronize();
    } else {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            BL_PROFILE("compute_bound"); // for NVIDIA Nsight compute
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real y = a(i,j,k);
                Real x = 1.0;
                for (int n = 0; n < 20; ++n) {
                    Real dx = -(x*x-y) / (2.*x);
                    x += dx;
                }
                a(i,j,k) = x;
            });
        }
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
        MultiFab mf(ba,dm,1,0);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            Array4<Real> const& a = mf.array(mfi);
            amrex::ParallelForRNG(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine)
            {
                a(i,j,k) = amrex::Random(engine) + 0.5;
            });
        }
        {
            BL_PROFILE("compute_bound-warmup");
            test(mf);
        }
        {
            BL_PROFILE("compute_bound-mf");
            amrex::Real t0 = amrex::second();
            test(mf);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}
