#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "Geometry/Surface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

JacobianCartesianToLocal::JacobianCartesianToLocal(const Surface& surface, 
			     const LocalTrajectoryParameters& localParameters) : theJacobian(5, 6, 0) {
 
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
  theJacobian(1,4) = -q*px/(p*p*p); theJacobian(1,5) = -q*py/(p*p*p); theJacobian(1,6) = -q*pz/(p*p*p);
  if(fabs(pz) > 0){
    theJacobian(2,4) = 1./pz;                                 theJacobian(2,6) = -px/(pz*pz);
                            theJacobian(3,5) = 1./pz;         theJacobian(3,6) = -py/(pz*pz);
  }
  theJacobian(4,1) = 1.;
  theJacobian(5,2) = 1.;
  LocalVector l1 = surface.toLocal(GlobalVector(1., 0., 0.));
  LocalVector l2 = surface.toLocal(GlobalVector(0., 1., 0.));
  LocalVector l3 = surface.toLocal(GlobalVector(0., 0., 1.));
  AlgebraicMatrix Rsub(3,3,0);
  Rsub(1,1) = l1.x(); Rsub(1,2) = l2.x(); Rsub(1,3) = l3.x();
  Rsub(2,1) = l1.y(); Rsub(2,2) = l2.y(); Rsub(2,3) = l3.y();
  Rsub(3,1) = l1.z(); Rsub(3,2) = l2.z(); Rsub(3,3) = l3.z();

  AlgebraicMatrix R(6,6,0);
  R.sub(1,1,Rsub);
  R.sub(4,4,Rsub);
  theJacobian = theJacobian*R;
}

const AlgebraicMatrix& JacobianCartesianToLocal::jacobian() const {
  return theJacobian;
}
