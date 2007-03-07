#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

JacobianLocalToCartesian::JacobianLocalToCartesian(const Surface& surface, 
			     const LocalTrajectoryParameters& localParameters) : theJacobian(6, 5, 0) {
  
  //LocalCoordinates 1 = (x, y, z, px, py, pz)
  //LocalCoordinates 2 = (q/|p|, dx/dz, dy/dz, x, y)
  //Transformation: x  = x
  //                y  = y
  //          px = (q/(q/|p|)) * (dx/dz) *  sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //          py = (q/(q/|p|)) * (dy/dz) * sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //          pz = (q/(q/|p|)) * sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //Jacobian J((x, y, px, py, pz)  = f(q/|p|, dx/dz, dy/dz, x, y))
  AlgebraicVector localTrackParams = localParameters.mixedFormatVector();
  double qbp = localTrackParams[0];
  double dxdz = localTrackParams[1];
  double dydz = localTrackParams[2];
  TrackCharge q = localParameters.charge();
  // for neutrals: qbp is 1/p instead of q/p - 
  //   equivalent to charge 1
  if ( q==0 )  q = 1;
  double pzSign = localParameters.pzSign();
  double sqr = sqrt(dxdz*dxdz + dydz*dydz + 1);
  
  theJacobian(1,4) = 1.;
  theJacobian(2,5) = 1.;
  theJacobian(4,1) = pzSign * ( (-q*dxdz)/(sqr*qbp*qbp) ); 
  theJacobian(4,2) = pzSign * ( q/(sqr*qbp) - (q*dxdz*dxdz)/(sqr*sqr*sqr*qbp) );
  theJacobian(4,3) = pzSign * ( (-q*dxdz*dydz)/(sqr*sqr*sqr*qbp) );
  theJacobian(5,1) = pzSign * ( (-q*dydz)/(sqr*qbp*qbp) );
  theJacobian(5,2) = pzSign * ( (-q*dxdz*dydz)/(sqr*sqr*sqr*qbp) );
  theJacobian(5,3) = pzSign * ( q/(sqr*qbp) - (q*dydz*dydz)/(sqr*sqr*sqr*qbp) );
  theJacobian(6,1) = pzSign * ( -q/(sqr*qbp*qbp) );
  theJacobian(6,2) = pzSign * ( (-q*dxdz)/(sqr*sqr*sqr*qbp) );
  theJacobian(6,3) = pzSign * ( (-q*dydz)/(sqr*sqr*sqr*qbp) );
  
  GlobalVector g1 = surface.toGlobal(LocalVector(1., 0., 0.));
  GlobalVector g2 = surface.toGlobal(LocalVector(0., 1., 0.));
  GlobalVector g3 = surface.toGlobal(LocalVector(0., 0., 1.));
  AlgebraicMatrix Rsub(3,3,0);
  Rsub(1,1) = g1.x(); Rsub(1,2) = g2.x(); Rsub(1,3) = g3.x();
  Rsub(2,1) = g1.y(); Rsub(2,2) = g2.y(); Rsub(2,3) = g3.y();
  Rsub(3,1) = g1.z(); Rsub(3,2) = g2.z(); Rsub(3,3) = g3.z();
  AlgebraicMatrix R(6,6,0);
  R.sub(1,1,Rsub);
  R.sub(4,4,Rsub);
  theJacobian = R * theJacobian;

}

const AlgebraicMatrix& JacobianLocalToCartesian::jacobian() const{
  return theJacobian;
}
