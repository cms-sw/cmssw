#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

JacobianCartesianToLocal::JacobianCartesianToLocal(const Surface& surface, 
			     const LocalTrajectoryParameters& localParameters) : theJacobian() {
 
  //LocalCoordinates 1 = (q/|p|, dx/dz, dy/dz, x, y)
  //LocalCoordinates 2 = (x, y, z, px, py, pz)
  //Transformation: q/|p| = q/sqrt(px*px + py*py + pz*pz)
  //                dx/dz = px/pz
  //                dy/dz = py/pz
  //                x     = x
  //                y     = y
  LocalVector plocal = localParameters.momentum();
  double px = plocal.x(), py = plocal.y(), pz = plocal.z();
  double p = plocal.mag();
  TrackCharge q = localParameters.charge();
  // for neutrals: qbp is 1/p instead of q/p - 
  //   equivalent to charge 1
  if ( q==0 )  q = 1;
  //Jacobian theJacobian( (q/|p|, dx/dz, dy/dz, x, y) = f(x, y, z, px, py, pz) )
  theJacobian(0,3) = -q*px/(p*p*p); theJacobian(0,4) = -q*py/(p*p*p); theJacobian(0,5) = -q*pz/(p*p*p);
  if(fabs(pz) > 0){
    theJacobian(1,3) = 1./pz;                                 theJacobian(1,5) = -px/(pz*pz);
                            theJacobian(2,4) = 1./pz;         theJacobian(2,5) = -py/(pz*pz);
  }
  theJacobian(3,0) = 1.;
  theJacobian(4,1) = 1.;

  /*
  LocalVector l1 = surface.toLocal(GlobalVector(1., 0., 0.));
  LocalVector l2 = surface.toLocal(GlobalVector(0., 1., 0.));
  LocalVector l3 = surface.toLocal(GlobalVector(0., 0., 1.));
  AlgebraicMatrix33 Rsub;
  Rsub(0,0) = l1.x(); Rsub(0,1) = l2.x(); Rsub(0,2) = l3.x();
  Rsub(1,0) = l1.y(); Rsub(1,1) = l2.y(); Rsub(1,2) = l3.y();
  Rsub(2,0) = l1.z(); Rsub(2,1) = l2.z(); Rsub(2,2) = l3.z();
  */
  
  AlgebraicMatrix33 Rsub;
  // need to be copied anhhow to go from float to double...
  Surface::RotationType const & rot = surface.rotation();
  Rsub(0,0) = rot.xx(); Rsub(0,1) = rot.xy(); Rsub(0,2) = rot.xz();
  Rsub(1,0) = rot.yx(); Rsub(1,1) = rot.yy(); Rsub(1,2) = rot.yz();
  Rsub(2,0) = rot.zx(); Rsub(2,1) = rot.zy(); Rsub(2,2) = rot.zz();



  AlgebraicMatrix66 R;
  R.Place_at(Rsub,0,0);
  R.Place_at(Rsub,3,3);
  theJacobian = theJacobian * R;
  //dbg::dbg_trace(1,"Ca2L", localParameters.vector(),l1,l2,l3,theJacobian);
}
const AlgebraicMatrix JacobianCartesianToLocal::jacobian_old() const {
  return asHepMatrix(theJacobian);
}
const AlgebraicMatrix56& JacobianCartesianToLocal::jacobian() const {
  return theJacobian;
}
