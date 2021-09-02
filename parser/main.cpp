#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

using namespace amrex;

static void test (MultiFab& mf, ParserExecutor<3> const& exe, Geometry const& geom)
{
    auto const problo = geom.ProbLoArray();
    auto const cellsize = geom.CellSizeArray();
    if (mf.isFusingCandidate()) {
        auto const& ma = mf.arrays();
        amrex::ParallelFor(mf,
        [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k) noexcept
        {
            Real x = (i+0.5)*cellsize[0] + problo[0];
            Real y = (j+0.5)*cellsize[1] + problo[1];
            Real z = (k+0.5)*cellsize[2] + problo[2];
            ma[bno](i,j,k) = exe(x,y,z);
        });
        Gpu::streamSynchronize();
    } else {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const Box& vbx = mfi.validbox();
            auto const& fab = mf.array(mfi);
            amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real x = (i+0.5)*cellsize[0] + problo[0];
                Real y = (j+0.5)*cellsize[1] + problo[1];
                Real z = (k+0.5)*cellsize[2] + problo[2];
                fab(i,j,k) = exe(x,y,z);
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
        Geometry geom(domain, RealBox({0.e-6, 0.0, -20.e-6}, {20.e-6, 1.e-10, 20.e-6}),
                      0, {0,0,0});

        MultiFab mf(ba,dm,1,0);

        Parser parser("epsilon/kp*2*x/w0**2*exp(-(x**2+y**2)/w0**2)*sin(k0*z)");
        parser.setConstant("epsilon",0.01);
        parser.setConstant("kp",3.5);
        parser.setConstant("w0",5.e-6);
        parser.setConstant("k0",3.e5);
        parser.registerVariables({"x","y","z"});
        auto const exe = parser.compile<3>();
        
        {
            BL_PROFILE("parser-warmup");
            test(mf, exe, geom);
        }
        {
            BL_PROFILE("parser");
            amrex::Real t0 = amrex::second();
            test(mf, exe, geom);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}
