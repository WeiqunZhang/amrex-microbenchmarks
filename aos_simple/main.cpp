#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace {
    constexpr int N = 27;
}

static void test (FabArray<BaseFab<GpuArray<Real,N> > >& stencil)
{
    if (stencil.isFusingCandidate()) {
        auto const& ma = stencil.arrays();
        amrex::ParallelFor(stencil,
        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) noexcept
        {
            for (int n = 0; n < N; ++n) {
                ma[b](i,j,k)[n] = -Real(n);
            }
        });
        Gpu::streamSynchronize();
    } else {
        for (MFIter mfi(stencil); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.validbox();
            auto const& aos = stencil.array(mfi);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int n = 0; n < N; ++n) {
                    aos(i,j,k)[n] = -Real(n);
                }
            });
        }
    }
}

int main (int argc, char* argv[])
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

        FabArray<BaseFab<GpuArray<Real,N> > > stencil(ba,dm,1,0);
        {
            BL_PROFILE("aos-warmup");
            test(stencil);
        }
        {
            BL_PROFILE("aos-mf");
            amrex::Real t0 = amrex::second();
            test(stencil);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}
