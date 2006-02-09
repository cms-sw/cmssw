#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "Geometry/Surface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

JacobianLocalToCurvilinear::
JacobianLocalToCurvilinear(const Surface& surface, 
			   const LocalTrajectoryParameters& localParameters,
			   const MagneticField& magField) : theJacobian(5, 5, 0) {
  
    // Origin: TRSDSC
  GlobalPoint  x = surface.toGlobal(localParameters.position());
  LocalVector tnl = localParameters.momentum().unit();
  GlobalVector dj = surface.toGlobal(LocalVector(1., 0., 0.));
  GlobalVector dk = surface.toGlobal(LocalVector(0., 1., 0.));
  GlobalVector tn = surface.toGlobal(tnl);

  GlobalVector p = surface.toGlobal(localParameters.momentum());
  GlobalVector pt(p.x(), p.y(), 0.);
  pt = pt.unit();
  //  GlobalVector di = tsos.surface().toGlobal(LocalVector(0., 0., 1.));

  // rotate coordinates because of wrong coordinate system in orca
  LocalVector tvw(tnl.z(), tnl.x(), tnl.y());
  double cosl = tn.perp(); if (cosl < 1.e-30) cosl = 1.e-30;
  double cosl1 = 1./cosl;
  GlobalVector un(-tn.y()*cosl1, tn.x()*cosl1, 0.);
  GlobalVector vn(-tn.z()*un.y(), tn.z()*un.x(), cosl);
  double uj = un.dot(dj);
  double uk = un.dot(dk);
  double vj = vn.dot(dj);
  double vk = vn.dot(dk);
  theJacobian(1,1) = 1.;
  theJacobian(2,2) = tvw.x()*vj;
  theJacobian(2,3) = tvw.x()*vk;
  theJacobian(3,2) = tvw.x()*uj*cosl1;
  theJacobian(3,3) = tvw.x()*uk*cosl1;
  theJacobian(4,4) = uj;
  theJacobian(4,5) = uk;
  theJacobian(5,4) = vj;
  theJacobian(5,5) = vk;
  // GlobalVector h = MagneticField::inInverseGeV(x);
  GlobalVector h  = magField.inTesla(x) * 2.99792458e-3;
  double q = -h.mag() * localParameters.signedInverseMomentum();
  double sinz =-un.dot(h.unit());
  double cosz = vn.dot(h.unit());
  theJacobian(2,4) = -q*tvw.y()*sinz;
  theJacobian(2,5) = -q*tvw.z()*sinz;
  theJacobian(3,4) = -q*tvw.y()*cosz*cosl1;
  theJacobian(3,5) = -q*tvw.z()*cosz*cosl1;
  // end of TRSDSC

}

const AlgebraicMatrix& JacobianLocalToCurvilinear::jacobian() const{
  return theJacobian;
}
