#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

JacobianLocalToCartesian::JacobianLocalToCartesian(const Surface& surface, 
			     const LocalTrajectoryParameters& localParameters) : theJacobian() {
  
  //LocalCoordinates 1 = (x, y, z, px, py, pz)
  //LocalCoordinates 2 = (q/|p|, dx/dz, dy/dz, x, y)
  //Transformation: x  = x
  //                y  = y
  //          px = (q/(q/|p|)) * (dx/dz) *  sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //          py = (q/(q/|p|)) * (dy/dz) * sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //          pz = (q/(q/|p|)) * sqrt(1./(1.+(dx/dz)^2+(dy/dz)^2))
  //Jacobian J((x, y, px, py, pz)  = f(q/|p|, dx/dz, dy/dz, x, y))

  // AlgebraicVector5 localTrackParams = localParameters.mixedFormatVector();
  double qbp = localParameters.qbp();
  double dxdz = localParameters.dxdz();
  double dydz = localParameters.dydz();
  TrackCharge iq = localParameters.charge();
  // for neutrals: qbp is 1/p instead of q/p - 
  //   equivalent to charge 1
  if ( iq==0 )  iq = 1;
  double pzSign = localParameters.pzSign();
  double q = iq*pzSign;
  double sqr = sqrt(dxdz*dxdz + dydz*dydz + 1);
  double den = -q/(sqr*sqr*sqr*qbp);

  // no difference between local and data member
  AlgebraicMatrix65 & lJacobian = theJacobian;
  lJacobian(0,3) = 1.;
  lJacobian(1,4) = 1.;
  lJacobian(3,0) = ( dxdz*(-q/(sqr*qbp*qbp)) ); 
  lJacobian(3,1) = ( q/(sqr*qbp) + (den*dxdz*dxdz) );
  lJacobian(3,2) = ( (den*dxdz*dydz) );
  lJacobian(4,0) = ( dydz*(-q/(sqr*qbp*qbp)) );
  lJacobian(4,1) = ( (den*dxdz*dydz) );
  lJacobian(4,2) = ( q/(sqr*qbp) + (den*dydz*dydz) );
  lJacobian(5,0) = ( -q/(sqr*qbp*qbp) );
  lJacobian(5,1) = ( (den*dxdz) );
  lJacobian(5,2) = ( (den*dydz) );
  
  /*
  GlobalVector g1 = surface.toGlobal(LocalVector(1., 0., 0.));
  GlobalVector g2 = surface.toGlobal(LocalVector(0., 1., 0.));
  GlobalVector g3 = surface.toGlobal(LocalVector(0., 0., 1.));
  */
  AlgebraicMatrix33 Rsub;
  /*
  Rsub(0,0) = g1.x(); Rsub(0,1) = g2.x(); Rsub(0,2) = g3.x();
  Rsub(1,0) = g1.y(); Rsub(1,1) = g2.y(); Rsub(1,2) = g3.y();
  Rsub(2,0) = g1.z(); Rsub(2,1) = g2.z(); Rsub(2,2) = g3.z();
  */
  // need to be copied anhhow to go from float to double...
  Surface::RotationType const & rot = surface.rotation();
  Rsub(0,0) = rot.xx(); Rsub(0,1) = rot.yx(); Rsub(0,2) = rot.zx();
  Rsub(1,0) = rot.xy(); Rsub(1,1) = rot.yy(); Rsub(1,2) = rot.zy();
  Rsub(2,0) = rot.xz(); Rsub(2,1) = rot.yz(); Rsub(2,2) = rot.zz();

  AlgebraicMatrix66 R;
  R.Place_at(Rsub, 0,0);
  R.Place_at(Rsub, 3,3);
  theJacobian = R * lJacobian;
  //dbg::dbg_trace(1,"Loc2Ca", localParameters.vector(),g1,g2,g3,theJacobian);
}

const AlgebraicMatrix65& JacobianLocalToCartesian::jacobian() const{
  return theJacobian;
}
const AlgebraicMatrix JacobianLocalToCartesian::jacobian_old() const{
  return asHepMatrix(theJacobian);
}
