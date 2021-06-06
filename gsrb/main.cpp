#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiMask.H>
#include <AMReX_BndryRegister.H>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void abec_gsrb (Box const& box, Array4<Real> const& phi, Array4<Real const> const& rhs,
                Real alpha, Array4<Real const> const& a,
                Real dhx, Real dhy, Real dhz,
                Array4<Real const> const& bX, Array4<Real const> const& bY,
                Array4<Real const> const& bZ,
                Array4<int const> const& m0, Array4<int const> const& m2,
                Array4<int const> const& m4,
                Array4<int const> const& m1, Array4<int const> const& m3,
                Array4<int const> const& m5,
                Array4<Real const> const& f0, Array4<Real const> const& f2,
                Array4<Real const> const& f4,
                Array4<Real const> const& f1, Array4<Real const> const& f3,
                Array4<Real const> const& f5,
                Box const& vbox, int redblack, int nc) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const auto vlo = amrex::lbound(vbox);
    const auto vhi = amrex::ubound(vbox);

    constexpr Real omega = Real(1.15);

    for (int n = 0; n < nc; ++n) {
        for         (int k = lo.z; k <= hi.z; ++k) {
            for     (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((i+j+k+redblack)%2 == 0) {
                        Real cf0 = (i == vlo.x && m0(vlo.x-1,j,k) > 0)
                            ? f0(vlo.x,j,k,n) : Real(0.0);
                        Real cf1 = (j == vlo.y && m1(i,vlo.y-1,k) > 0)
                            ? f1(i,vlo.y,k,n) : Real(0.0);
                        Real cf2 = (k == vlo.z && m2(i,j,vlo.z-1) > 0)
                            ? f2(i,j,vlo.z,n) : Real(0.0);
                        Real cf3 = (i == vhi.x && m3(vhi.x+1,j,k) > 0)
                            ? f3(vhi.x,j,k,n) : Real(0.0);
                        Real cf4 = (j == vhi.y && m4(i,vhi.y+1,k) > 0)
                            ? f4(i,vhi.y,k,n) : Real(0.0);
                        Real cf5 = (k == vhi.z && m5(i,j,vhi.z+1) > 0)
                            ? f5(i,j,vhi.z,n) : Real(0.0);

                        Real gamma = alpha*a(i,j,k)
                            +   dhx*(bX(i,j,k,n)+bX(i+1,j,k,n))
                            +   dhy*(bY(i,j,k,n)+bY(i,j+1,k,n))
                            +   dhz*(bZ(i,j,k,n)+bZ(i,j,k+1,n));

                        Real g_m_d = gamma
                            - (dhx*(bX(i,j,k,n)*cf0 + bX(i+1,j,k,n)*cf3)
                            +  dhy*(bY(i,j,k,n)*cf1 + bY(i,j+1,k,n)*cf4)
                            +  dhz*(bZ(i,j,k,n)*cf2 + bZ(i,j,k+1,n)*cf5));

                        Real rho =  dhx*( bX(i  ,j,k,n)*phi(i-1,j,k,n)
                                  +       bX(i+1,j,k,n)*phi(i+1,j,k,n) )
                                  + dhy*( bY(i,j  ,k,n)*phi(i,j-1,k,n)
                                  +       bY(i,j+1,k,n)*phi(i,j+1,k,n) )
                                  + dhz*( bZ(i,j,k  ,n)*phi(i,j,k-1,n)
                                  +       bZ(i,j,k+1,n)*phi(i,j,k+1,n) );

                        Real res =  rhs(i,j,k,n) - (gamma*phi(i,j,k,n) - rho);
                        phi(i,j,k,n) = phi(i,j,k,n) + omega/g_m_d * res;
                    }
                }
            }
        }
    }
}

}

static void test (MultiFab& sol, MultiFab const& rhs, MultiFab const& acoef,
                  MultiFab const& bxcoef, MultiFab const& bycoef, MultiFab const& bzcoef,
                  BndryRegister const& undrrelxr,
                  Array<MultiMask,2*AMREX_SPACEDIM> const& maskvals)
{
    OrientationIter oitr;

    const Real alpha = 1.0;
    const Real dhx = 1.0e3;
    const Real dhy = 1.0e3;
    const Real dhz = 1.0e3;
    const int redblack = 1;
    const int nc = 1;

    const FabSet& f0 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f1 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f2 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f3 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f4 = undrrelxr[oitr()]; ++oitr;
    const FabSet& f5 = undrrelxr[oitr()]; ++oitr;

    const MultiMask& mm0 = maskvals[0];
    const MultiMask& mm1 = maskvals[1];
    const MultiMask& mm2 = maskvals[2];
    const MultiMask& mm3 = maskvals[3];
    const MultiMask& mm4 = maskvals[4];
    const MultiMask& mm5 = maskvals[5];

    for (MFIter mfi(sol,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto& m0 = mm0.array(mfi);
        const auto& m1 = mm1.array(mfi);
        const auto& m2 = mm2.array(mfi);
        const auto& m3 = mm3.array(mfi);
        const auto& m4 = mm4.array(mfi);
        const auto& m5 = mm5.array(mfi);

        const Box& tbx = mfi.tilebox();
        const Box& vbx = mfi.validbox();
        const auto& solnfab = sol.array(mfi);
        const auto& rhsfab  = rhs.array(mfi);
        const auto& afab    = acoef.array(mfi);

        const auto& bxfab = bxcoef.array(mfi);
        const auto& byfab = bycoef.array(mfi);
        const auto& bzfab = bzcoef.array(mfi);

        const auto& f0fab = f0.array(mfi);
        const auto& f1fab = f1.array(mfi);
        const auto& f2fab = f2.array(mfi);
        const auto& f3fab = f3.array(mfi);
        const auto& f4fab = f4.array(mfi);
        const auto& f5fab = f5.array(mfi);

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA ( tbx, thread_box,
        {
            abec_gsrb(thread_box, solnfab, rhsfab, alpha, afab,
                      AMREX_D_DECL(dhx, dhy, dhz),
                      AMREX_D_DECL(bxfab, byfab, bzfab),
                      AMREX_D_DECL(m0,m2,m4),
                      AMREX_D_DECL(m1,m3,m5),
                      AMREX_D_DECL(f0fab,f2fab,f4fab),
                      AMREX_D_DECL(f1fab,f3fab,f5fab),
                      vbx, redblack, nc);
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

        Geometry geom(Box(IntVect(0),IntVect(255)),
                      RealBox(0., 0., 0., 1., 1., 1.),
                      0, {0,0,0});

        BoxList bl;
        bl.push_back(Box(IntVect(0,0,0),IntVect(127,127,127)));
        bl.push_back(Box(IntVect(128,64,64),IntVect(255,191,191)));
        BoxArray ba(bl);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        MultiFab sol(ba,dm,1,1);
        sol.setVal(1.0);

        MultiFab rhs(ba,dm,1,0);
        rhs.setVal(2.0);

        MultiFab acoef(ba,dm,1,0);
        acoef.setVal(1.0);

        MultiFab bxcoef(amrex::convert(ba,IntVect(1,0,0)), dm, 1, 0);
        bxcoef.setVal(3.1);
        MultiFab bycoef(amrex::convert(ba,IntVect(0,1,0)), dm, 1, 0);
        bycoef.setVal(3.2);
        MultiFab bzcoef(amrex::convert(ba,IntVect(0,0,1)), dm, 1, 0);
        bzcoef.setVal(3.3);

        BndryRegister undrrelxr(ba,dm,1,0,0,1);
        undrrelxr.setVal(0.5);

        Array<MultiMask,2*AMREX_SPACEDIM> maskvals;
        for (OrientationIter oitr; oitr; ++oitr) {
            maskvals[oitr()].define(ba,dm,geom,oitr(),0,1,0,1,true);
        }

        {
            BL_PROFILE("gsrb-warmup");
            test(sol,rhs,acoef,bxcoef,bycoef,bzcoef,undrrelxr,maskvals);
        }
        {
            BL_PROFILE("gsrb");
            amrex::Real t0 = amrex::second();
            test(sol,rhs,acoef,bxcoef,bycoef,bzcoef,undrrelxr,maskvals);
            t = amrex::second()-t0;
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t << ".\n";
}