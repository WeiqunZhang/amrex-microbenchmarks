#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Scan.H>

using namespace amrex;

static void test (iMultiFab& mfa, iMultiFab& mfb)
{
    for (MFIter mfi(mfa); mfi.isValid(); ++mfi) {
        Scan::ExclusiveSum(mfi.fabbox().numPts(), mfb[mfi].dataPtr(), mfa[mfi].dataPtr());
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
        iMultiFab mfa(ba,dm,1,0);
        iMultiFab mfb(ba,dm,1,0);
        mfa.setVal(1);
        mfb.setVal(1);
        {
            BL_PROFILE("scan-warmup");
            test(mfa,mfb);
        }
        {
            BL_PROFILE("scan-mf");
            amrex::Real t0 = amrex::second();
            test(mfa,mfb);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}
