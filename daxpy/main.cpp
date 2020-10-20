#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        Box domain(IntVect(0),IntVect(255));
        {
            BoxArray ba(domain);
            DistributionMapping dm{ba};
            MultiFab mfa(ba,dm,1,0);
            MultiFab mfb(ba,dm,1,0);
            mfa.setVal(1.0);
            mfb.setVal(2.0);
            MultiFab::Saxpy(mfa,0.5,mfb,0,0,1,0);
            {
                BL_PROFILE_REGION("daxpy-256");
                MultiFab::Saxpy(mfa,0.5,mfb,0,0,1,0);
            }
        }
        {
            BoxArray ba(domain);
            ba.maxSize(64);
            DistributionMapping dm{ba};
            MultiFab mfa(ba,dm,1,0);
            MultiFab mfb(ba,dm,1,0);
            mfa.setVal(1.0);
            mfb.setVal(2.0);
            MultiFab::Saxpy(mfa,0.5,mfb,0,0,1,0);
            {
                BL_PROFILE_REGION("daxpy-64");
                MultiFab::Saxpy(mfa,0.5,mfb,0,0,1,0);
            }
        }
    }
    amrex::Finalize();
}
