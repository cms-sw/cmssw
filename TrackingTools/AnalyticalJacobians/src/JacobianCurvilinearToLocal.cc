#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "MagneticField/Engine/interface/MagneticField.h"

JacobianCurvilinearToLocal::
JacobianCurvilinearToLocal(const Surface& surface, 
			   const LocalTrajectoryParameters& localParameters,
			   const MagneticField& magField) : theJacobian() {
 
  GlobalPoint  x = surface.toGlobal(localParameters.position());
  GlobalVector h  = magField.inInverseGeV(x);
  GlobalVector qh = h*localParameters.signedInverseMomentum();  // changed sign


  //  GlobalVector  hdir =  h.unit();
  //double q = -h.mag() * localParameters.signedInverseMomentum();
  
  LocalVector tnl = localParameters.direction();
  GlobalVector tn = surface.toGlobal(tnl);
  double t1r = 1./tnl.z();

  // GlobalVector dj = surface.toGlobal(LocalVector(1., 0., 0.));
  // GlobalVector dk = surface.toGlobal(LocalVector(0., 1., 0.));
  //  GlobalVector di = surface.toGlobal(LocalVector(0., 0., 1.));
  Surface::RotationType const & rot = surface.rotation();

  compute(rot, tn, qh, t1r);
}

JacobianCurvilinearToLocal::
JacobianCurvilinearToLocal(const Surface& surface, 
			   const LocalTrajectoryParameters& localParameters,
			   const GlobalTrajectoryParameters& globalParameters,
			   const MagneticField& magField) : theJacobian() {
 
  GlobalPoint  x =  globalParameters.position();
  GlobalVector h  = magField.inInverseGeV(x);
  GlobalVector qh = h*localParameters.signedInverseMomentum();  // changed sign

  //GlobalVector  hdir =  h.unit();
  //double q = -h.mag() * localParameters.signedInverseMomentum();

 
  //  GlobalVector tn = globalParameters.momentum().unit();
  //  LocalVector tnl = localParameters.momentum().unit();

  LocalVector tnl = localParameters.direction();
  // GlobalVector tn = surface.toGlobal(tnl); // faster?
  GlobalVector tn =  globalParameters.momentum()*std::abs(localParameters.signedInverseMomentum());
  double t1r = 1./tnl.z();
 
 
  Surface::RotationType const & rot = surface.rotation();

  compute(rot, tn, qh, t1r);
}


void JacobianCurvilinearToLocal::compute(Surface::RotationType const & rot, GlobalVector  const & tn, GlobalVector const & qh, double t1r) {
  // Origin: TRSCSD

  double cosl = tn.perp(); if (cosl < 1.e-30) cosl = 1.e-30;
  double cosl1 = 1./cosl;
  GlobalVector un(-tn.y()*cosl1, tn.x()*cosl1, 0.);
  GlobalVector vn(-tn.z()*un.y(), tn.z()*un.x(), cosl);

  auto u = rot.rotate(un.basicVector());
  auto v = rot.rotate(vn.basicVector());

  int j=0, k=1, i=2;

  //  double t1r = 1./tvw.x();
  double t2r = t1r*t1r;
  double t3r = t1r*t2r;

  theJacobian(0,0) = 1.;
  theJacobian(1,1) = -u[k]*t2r;
  theJacobian(1,2) = v[k]*(cosl*t2r);
  theJacobian(2,1) = u[j]*t2r;
  theJacobian(2,2) = -v[j]*(cosl*t2r);
  theJacobian(3,3) = v[k]*t1r;
  theJacobian(3,4) = -u[k]*t1r;
  theJacobian(4,3) = -v[j]*t1r;
  theJacobian(4,4) = u[j]*t1r;


  double sinz = un.dot(qh);
  double cosz =-vn.dot(qh);
  double ui = u[i]*(t3r);
  double vi = v[i]*(t3r);
  theJacobian(1,3) =-ui*(v[k]*cosz-u[k]*sinz);
  theJacobian(1,4) =-vi*(v[k]*cosz-u[k]*sinz);
  theJacobian(2,3) = ui*(v[j]*cosz-u[j]*sinz);
  theJacobian(2,4) = vi*(v[j]*cosz-u[j]*sinz);
  // end of TRSCSD
  //dbg::dbg_trace(1,"Cu2L", localParameters.vector(),di,dj,dk,theJacobian);
}
