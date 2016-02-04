#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

JacobianLocalToCurvilinear::
JacobianLocalToCurvilinear(const Surface& surface, 
			   const LocalTrajectoryParameters& localParameters,
			   const MagneticField& magField) : theJacobian() {
  
  // Origin: TRSDSC

  GlobalPoint  x = surface.toGlobal(localParameters.position());
  GlobalVector h  = magField.inInverseGeV(x);

 
  LocalVector tnl = localParameters.momentum().unit();
  GlobalVector tn = surface.toGlobal(tnl);
 
  // GlobalVector dj = surface.toGlobal(LocalVector(1., 0., 0.));
  // GlobalVector dk = surface.toGlobal(LocalVector(0., 1., 0.));
  Surface::RotationType const & rot = surface.rotation();
  GlobalVector dj(rot.x());
  GlobalVector dk(rot.y());
 
  // GlobalVector p = surface.toGlobal(localParameters.momentum());
  // GlobalVector pt(p.x(), p.y(), 0.);
  // pt = pt.unit();
  // GlobalVector tn = p.unit();

  //  GlobalVector di = tsos.surface().toGlobal(LocalVector(0., 0., 1.));

  // rotate coordinates because of wrong coordinate system in orca
  LocalVector tvw(tnl.z(), tnl.x(), tnl.y());
  double cosl = tn.perp(); if (cosl < 1.e-30) cosl = 1.e-30;
  double cosl1 = 1./cosl;

  double q = -h.mag() * localParameters.signedInverseMomentum();

  GlobalVector un(-tn.y()*cosl1, tn.x()*cosl1, 0.);
  double uj = un.dot(dj);
  double uk = un.dot(dk);
  double sinz =-un.dot(h.unit());

  GlobalVector vn(-tn.z()*un.y(), tn.z()*un.x(), cosl);
  double vj = vn.dot(dj);
  double vk = vn.dot(dk);
  double cosz = vn.dot(h.unit());
 

  theJacobian(0,0) = 1.;
  theJacobian(1,1) = tvw.x()*vj;
  theJacobian(1,2) = tvw.x()*vk;
  theJacobian(2,1) = tvw.x()*uj*cosl1;
  theJacobian(2,2) = tvw.x()*uk*cosl1;
  theJacobian(3,3) = uj;
  theJacobian(3,4) = uk;
  theJacobian(4,3) = vj;
  theJacobian(4,4) = vk;
 
  theJacobian(1,3) = -q*tvw.y()*sinz;
  theJacobian(1,4) = -q*tvw.z()*sinz;
  theJacobian(2,3) = -q*tvw.y()*(cosz*cosl1);
  theJacobian(2,4) = -q*tvw.z()*(cosz*cosl1);
  // end of TRSDSC

  //dbg::dbg_trace(1,"Loc2Cu", localParameters.vector(),x,dj,dk,theJacobian);
}
const AlgebraicMatrix JacobianLocalToCurvilinear::jacobian_old() const{
  return asHepMatrix(theJacobian);
}
const AlgebraicMatrix55& JacobianLocalToCurvilinear::jacobian() const{
  return theJacobian;
}
