#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public MagneticField {
    M5T() : m(0., 0., 5.) {}
    GlobalVector inTesla(const GlobalPoint&) const override { return m; }

    GlobalVector m;
  };

}  // namespace

#include "FWCore/Utilities/interface/HRRealTime.h"
void st() {}
void en() {}

#include <iostream>

int main() {
  M5T const m;

  Basic3DVector<float> axis(0.5, 1., 1);

  Surface::RotationType rot(axis, 0.5 * M_PI);
  std::cout << rot << std::endl;

  Surface::PositionType pos(0., 0., 0.);

  Plane plane(pos, rot);
  LocalTrajectoryParameters tpl(-1. / 3.5, 1., 1., 0., 0., 1.);
  GlobalVector mg = plane.toGlobal(tpl.momentum());
  GlobalTrajectoryParameters tpg(pos, mg, -1., &m);
  std::cout << tpl.position() << " " << tpl.momentum() << std::endl;
  std::cout << tpg.position() << " " << tpg.momentum() << std::endl;

  double curv = tpg.transverseCurvature();

  std::cout << curv << " " << mg.mag() << std::endl;

  AlgebraicMatrix55 fullJacobian = AlgebraicMatrixID();
  AlgebraicMatrix55 deltaJacobian = AlgebraicMatrixID();
  GlobalTrajectoryParameters tpg0(pos, mg, -1., &m);
  double totalStep(0.);
  double singleStep(1.);
  for (int i = 0; i < 10; ++i) {
    HelixForwardPlaneCrossing prop(HelixForwardPlaneCrossing::PositionType(tpg.position()),
                                   HelixForwardPlaneCrossing::DirectionType(tpg.momentum()),
                                   curv);
    GlobalVector h = tpg.magneticFieldInInverseGeV(tpg.position());
    double s = singleStep;
    totalStep += singleStep;
    GlobalPoint x(prop.position(s));
    GlobalVector p(prop.direction(s));
    GlobalTrajectoryParameters tpg2(x, p.unit(), curv, 0, &m);
    std::cout << x << " " << p << std::endl;
    std::cout << tpg2.position() << " " << tpg2.momentum() << std::endl;
    AnalyticalCurvilinearJacobian full;
    AnalyticalCurvilinearJacobian delta;
    full.computeFullJacobian(tpg, tpg2.position(), tpg2.momentum(), h, s);
    std::cout << full.jacobian() << std::endl;
    std::cout << std::endl;
    delta.computeInfinitesimalJacobian(tpg, x, tpg2.momentum(), h, s);
    std::cout << delta.jacobian() << std::endl;
    std::cout << std::endl;
    tpg = tpg2;
    fullJacobian *= full.jacobian();
    deltaJacobian *= delta.jacobian();
  }
  std::cout << "---------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << fullJacobian << std::endl;
  std::cout << std::endl;
  std::cout << deltaJacobian << std::endl;
  std::cout << std::endl;
  AnalyticalCurvilinearJacobian full;
  GlobalVector h = tpg0.magneticFieldInInverseGeV(tpg0.position());
  full.computeFullJacobian(tpg0, tpg.position(), tpg.momentum(), h, totalStep);
  std::cout << full.jacobian() << std::endl;
  std::cout << std::endl;

  AlgebraicMatrix55 div;
  for (unsigned int i = 0; i < 5; ++i) {
    for (unsigned int j = 0; j < 5; ++j) {
      //       div(i,j) = fabs(full.jacobian()(i,j))>1.e-20 ? fullJacobian(i,j)/full.jacobian()(i,j)-1 : 0;
      div(i, j) = fullJacobian(i, j) - full.jacobian()(i, j);
    }
  }
  std::cout << "Full relative" << std::endl << div << std::endl;
  for (unsigned int i = 0; i < 5; ++i) {
    for (unsigned int j = 0; j < 5; ++j) {
      //       div(i,j) = fabs(full.jacobian()(i,j))>1.e-20 ? deltaJacobian(i,j)/full.jacobian()(i,j)-1 : 0;
      div(i, j) = deltaJacobian(i, j) - full.jacobian()(i, j);
    }
  }
  std::cout << "Delta relative" << std::endl << div << std::endl;

  //
  // plane at end of propagation (equivalent to Curvilinear definition up to rotation in the plane)
  //
  Basic3DVector<float> newaxis(tpg.momentum().unit());
  Surface::RotationType newrot(axis, 0.);
  Surface::PositionType newpos(tpg.position());
  Plane newplane(newpos, newrot);
  //
  // total jacobian (local to local)
  //
  JacobianLocalToCurvilinear jlc(plane, tpl, m);
  LocalTrajectoryParameters newlpg(newplane.toLocal(tpg.position()), newplane.toLocal(tpg.momentum()), 1.);
  JacobianCurvilinearToLocal jcl(newplane, newlpg, m);
  //   AlgebraicMatrix55 jacobianL2L = jcl.jacobian()*full.jacobian()*jlc.jacobian();
  AlgebraicMatrix55 jacobianL2L = jcl.jacobian() * deltaJacobian * jlc.jacobian();
  //
  // redo propagation to target plane (should give identical position ...)
  //
  HelixArbitraryPlaneCrossing aprop(HelixForwardPlaneCrossing::PositionType(tpg0.position()),
                                    HelixForwardPlaneCrossing::DirectionType(tpg0.momentum()),
                                    curv);
  std::pair<bool, double> as = aprop.pathLength(newplane);
  std::cout << "as = " << as.second << std::endl;
  GlobalPoint ax(aprop.position(as.second));
  GlobalVector ap(aprop.direction(as.second) * tpg0.momentum().perp());
  //
  // change initial q/p and redo propagation
  //
  double qpmod = 1. / tpg0.momentum().mag() + 0.01;
  GlobalTrajectoryParameters tpg0mod(tpg0.position(), tpg0.momentum().unit() / fabs(qpmod), qpmod > 0 ? 1. : -1., &m);
  HelixArbitraryPlaneCrossing aprop2(HelixForwardPlaneCrossing::PositionType(tpg0mod.position()),
                                     HelixForwardPlaneCrossing::DirectionType(tpg0mod.momentum()),
                                     tpg0mod.transverseCurvature());
  std::pair<bool, double> as2 = aprop2.pathLength(newplane);
  std::cout << "as2 = " << as2.second << std::endl;
  GlobalPoint ax2(aprop2.position(as2.second));
  GlobalVector ap2(aprop2.direction(as2.second) * tpg0mod.momentum().perp());
  //
  // local coordinates after 2nd propagation
  LocalVector newDx = newplane.toLocal(ax2 - ax);
  LocalVector newDp = newplane.toLocal(ap2 / ap2.z() - ap / ap.z());
  std::cout << newDx << std::endl;
  std::cout << newDp << std::endl;
  //
  // prediction of variations with jacobian
  //
  AlgebraicVector5 dp;
  dp(0) = qpmod - 1. / tpg0.momentum().mag();
  dp(1) = dp(2) = dp(3) = dp(4) = 0;
  AlgebraicVector5 dpPred = jacobianL2L * dp;
  std::cout << dpPred << std::endl;
  //
  // comparison with difference in parameters
  //
  std::cout << newDx.x() - dpPred[3] << " " << newDx.y() - dpPred[4] << std::endl;
  std::cout << newDp.x() - dpPred[1] << " " << newDp.y() - dpPred[2] << std::endl;
  std::cout << (1. / ap2.mag() - 1. / ap.mag()) - dpPred[0] << std::endl;

  return 0;
}
