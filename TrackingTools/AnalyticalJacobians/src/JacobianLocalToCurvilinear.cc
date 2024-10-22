#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "MagneticField/Engine/interface/MagneticField.h"

JacobianLocalToCurvilinear::JacobianLocalToCurvilinear(const Surface& surface,
                                                       const LocalTrajectoryParameters& localParameters,
                                                       const MagneticField& magField)
    : theJacobian(ROOT::Math::SMatrixNoInit()) {
  GlobalPoint x = surface.toGlobal(localParameters.position());
  GlobalVector h = magField.inInverseGeV(x);
  GlobalVector hq = h * localParameters.signedInverseMomentum();  // changed sign

  LocalVector tnl = localParameters.direction();
  GlobalVector tn = surface.toGlobal(tnl);

  Surface::RotationType const& rot = surface.rotation();

  compute(rot, tnl, tn, hq);
}

JacobianLocalToCurvilinear::JacobianLocalToCurvilinear(const Surface& surface,
                                                       const LocalTrajectoryParameters& localParameters,
                                                       const GlobalTrajectoryParameters& globalParameters,
                                                       const MagneticField& magField)
    : theJacobian(ROOT::Math::SMatrixNoInit()) {
  GlobalVector h = globalParameters.magneticFieldInInverseGeV();
  GlobalVector hq = h * localParameters.signedInverseMomentum();  // changed sign

  LocalVector tnl = localParameters.direction();
  GlobalVector tn = surface.toGlobal(tnl);  // globalParameters.momentum().unit();

  Surface::RotationType const& rot = surface.rotation();

  compute(rot, tnl, tn, hq);
}

void JacobianLocalToCurvilinear::compute(Surface::RotationType const& rot,
                                         LocalVector const& tnl,
                                         GlobalVector const& tn,
                                         GlobalVector const& hq) {
  // Origin: TRSDSC

  GlobalVector dj(rot.x());
  GlobalVector dk(rot.y());

  // rotate coordinates because of wrong coordinate system in orca
  double tvwX = tnl.z(), tvwY = tnl.x(), tvwZ = tnl.y();
  double cosl = tn.perp();
  if (cosl < 1.e-30)
    cosl = 1.e-30;
  double cosl1 = 1. / cosl;

  GlobalVector un(-tn.y() * cosl1, tn.x() * cosl1, 0.);
  double uj = un.dot(dj);
  double uk = un.dot(dk);
  double sinz = -un.dot(hq);

  GlobalVector vn(-tn.z() * un.y(), tn.z() * un.x(), cosl);
  double vj = vn.dot(dj);
  double vk = vn.dot(dk);
  double cosz = vn.dot(hq);

  theJacobian(0, 0) = 1.;
  for (auto i = 1; i < 5; ++i)
    theJacobian(0, i) = 0.;

  theJacobian(1, 0) = 0.;
  theJacobian(2, 0) = 0.;

  theJacobian(1, 1) = tvwX * vj;
  theJacobian(1, 2) = tvwX * vk;
  theJacobian(2, 1) = tvwX * uj * cosl1;
  theJacobian(2, 2) = tvwX * uk * cosl1;

  for (auto i = 0; i < 3; ++i) {
    theJacobian(3, i) = 0.;
    theJacobian(4, i) = 0.;
  }

  theJacobian(3, 3) = uj;
  theJacobian(3, 4) = uk;
  theJacobian(4, 3) = vj;
  theJacobian(4, 4) = vk;

  theJacobian(1, 3) = tvwY * sinz;
  theJacobian(1, 4) = tvwZ * sinz;
  theJacobian(2, 3) = tvwY * (cosz * cosl1);
  theJacobian(2, 4) = tvwZ * (cosz * cosl1);
  // end of TRSDSC

  //dbg::dbg_trace(1,"Loc2Cu", localParameters.vector(),x,dj,dk,theJacobian);
}
