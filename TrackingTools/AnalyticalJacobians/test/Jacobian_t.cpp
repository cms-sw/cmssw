#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public MagneticField {
    M5T() : m(0., 0., 5.) {}
    virtual GlobalVector inTesla(const GlobalPoint&) const { return m; }

    GlobalVector m;
  };

}  // namespace

#include "FWCore/Utilities/interface/HRRealTime.h"
void st(void* ptr) {
  asm("" /* no instructions */
      :  /* no inputs */
      : /* output */ "rax"(ptr)
      : /* pretend to clobber */ "memory");
}
void en(void const* ptr) {
  asm("" /* no instructions */
      :  /* no inputs */
      : /* output */ "rax"(ptr)
      : /* pretend to clobber */ "memory");
}

int main() {
  // GlobalVector xx(0.5,1.,1.);
  // GlobalVector yy(-1.,0.5,1.);

  Basic3DVector<float> axis(0.5, 1., 1);

  Surface::RotationType rot(axis, 0.5 * M_PI);
  std::cout << rot << std::endl;

  Surface::PositionType pos(0., 0., 0.);

  Plane plane(pos, rot);

  GlobalVector g1 = plane.toGlobal(LocalVector(1., 0., 0.));
  GlobalVector g2 = plane.toGlobal(LocalVector(0., 1., 0.));
  GlobalVector g3 = plane.toGlobal(LocalVector(0., 0., 1.));
  AlgebraicMatrix33 Rsub;
  Rsub(0, 0) = g1.x();
  Rsub(0, 1) = g2.x();
  Rsub(0, 2) = g3.x();
  Rsub(1, 0) = g1.y();
  Rsub(1, 1) = g2.y();
  Rsub(1, 2) = g3.y();
  Rsub(2, 0) = g1.z();
  Rsub(2, 1) = g2.z();
  Rsub(2, 2) = g3.z();

  std::cout << Rsub << std::endl;

  if (rot.xx() != Rsub(0, 0) || rot.xy() != Rsub(1, 0) || rot.xz() != Rsub(2, 0) || rot.yx() != Rsub(0, 1) ||
      rot.yy() != Rsub(1, 1) || rot.yz() != Rsub(2, 1) || rot.zx() != Rsub(0, 2) || rot.zy() != Rsub(1, 2) ||
      rot.zz() != Rsub(2, 2))
    std::cout << " wrong assumption!" << std::endl;

  GlobalVector dj(rot.x());
  GlobalVector dk(rot.y());
  GlobalVector di(rot.z());

  GlobalVector un(-1., 1.5, 0.5);
  double ui = un.dot(di);
  double uj = un.dot(dj);
  double uk = un.dot(dk);
  std::cout << '\n' << un << std::endl;
  std::cout << '\n' << uj << "," << uk << "," << ui << std::endl;
  std::cout << rot.rotate(un.basicVector()) << std::endl;
  std::cout << rot.rotateBack(un.basicVector()) << '\n' << std::endl;

  M5T const m;
  LocalTrajectoryParameters tp(1., 1., 1., 0., 0., 1.);
  std::cout << tp.vector() << std::endl;
  std::cout << tp.charge() << " " << tp.signedInverseMomentum() << std::endl;
  GlobalVector tn = plane.toGlobal(tp.momentum()).unit();
  GlobalTrajectoryParameters gp(plane.toGlobal(tp.position()), plane.toGlobal(tp.momentum()), tp.charge(), &m);
  LocalVector tnl = plane.toLocal(tn);
  std::cout << tp.position() << std::endl;
  std::cout << tp.momentum() << std::endl;
  std::cout << tp.momentum().unit() << std::endl;
  std::cout << tp.direction() << std::endl;
  std::cout << tnl << std::endl;
  std::cout << tn << std::endl;
  std::cout << plane.toGlobal(tnl) << std::endl;
  std::cout << gp.direction() << std::endl;

  // verify cart to curv and back....
  // AlgebraicMatrixID();

  int N = 1000000;

  // L =>Cart
  {
    std::cout << "L =>Cart ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianLocalToCartesian __attribute__((aligned(16))) jl2c(plane, tp);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianLocalToCartesian __attribute__((aligned(16))) jl2c(plane, tp);
    std::cout << jl2c.jacobian() << std::endl;
  }

  // L => Curv
  {
    std::cout << "L =>Curv from loc ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianLocalToCurvilinear __attribute__((aligned(16))) jl2c(plane, tp, m);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianLocalToCurvilinear __attribute__((aligned(16))) jl2c(plane, tp, m);
    std::cout << jl2c.jacobian() << std::endl;
  }

  {
    std::cout << "L =>Curv from loc+glob ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianLocalToCurvilinear __attribute__((aligned(16))) jl2c(plane, tp, gp, m);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianLocalToCurvilinear __attribute__((aligned(16))) jl2c(plane, tp, gp, m);
    std::cout << jl2c.jacobian() << std::endl;
  }

  // Cart => Loc
  {
    std::cout << "Cart => Loc ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianCartesianToLocal __attribute__((aligned(16))) jl2c(plane, tp);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianCartesianToLocal __attribute__((aligned(16))) jl2c(plane, tp);
    std::cout << jl2c.jacobian() << std::endl;
  }

  // Curv => Loc
  {
    std::cout << "Curv => Loc from loc ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianCurvilinearToLocal __attribute__((aligned(16))) jl2c(plane, tp, m);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianCurvilinearToLocal __attribute__((aligned(16))) jl2c(plane, tp, m);
    std::cout << jl2c.jacobian() << std::endl;
  }

  {
    std::cout << "Curv => Loc from loc + glob ";
    edm::HRTimeType s = edm::hrRealTime();
    for (int i = 0; i < N; ++i) {
      st(&tp);
      JacobianCurvilinearToLocal __attribute__((aligned(16))) jl2c(plane, tp, gp, m);
      en(&jl2c.jacobian());
    }
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << double(e - s) / N << std::endl;
    JacobianCurvilinearToLocal __attribute__((aligned(16))) jl2c(plane, tp, gp, m);
    std::cout << jl2c.jacobian() << std::endl;
  }

  return 0;
}
