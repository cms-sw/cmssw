#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

JacobianCurvilinearToCartesian::
JacobianCurvilinearToCartesian(const GlobalTrajectoryParameters& globalParameters) : theJacobian(6, 5, 0) {
  GlobalVector xt = globalParameters.momentum();
  //GlobalVector yt(xt.y(), -xt.x(), 0.); \\wrong direction of the axis
  //GlobalVector zt(xt.x()*xt.z(), xt.y()*xt.z(), -xt.perp2()); \\and then also on this one
  GlobalVector yt(-xt.y(), xt.x(), 0.); 
  GlobalVector zt = xt.cross(yt);

  GlobalVector pvec = globalParameters.momentum();
  double pt = pvec.perp();
  TrackCharge q = globalParameters.charge();
  // for neutrals: qbp is 1/p instead of q/p - 
  //   equivalent to charge 1
  if ( q==0 )  q = 1;

  xt = xt.unit(); 
  if(fabs(pt) > 0){
    yt = yt.unit(); 
    zt = zt.unit();
  }
  
  AlgebraicMatrix R(6,6,0);
  R(1,1) = xt.x(); R(1,2) = yt.x(); R(1,3) = zt.x();
  R(2,1) = xt.y(); R(2,2) = yt.y(); R(2,3) = zt.y();
  R(3,1) = xt.z(); R(3,2) = yt.z(); R(3,3) = zt.z();
  R(4,4) = 1.;
  R(5,5) = 1.;
  R(6,6) = 1.;

  double p = pvec.mag(), p2 = p*p;
  double lambda = 0.5 * M_PI - pvec.theta();
  double phi = pvec.phi();
  double sinlambda = sin(lambda), coslambda = cos(lambda);
  double sinphi = sin(phi), cosphi = cos(phi);

  theJacobian(2,4) = 1.;
  theJacobian(3,5) = 1.;
  theJacobian(4,1) = -q * p2 * coslambda * cosphi;
  theJacobian(4,2) = -p * sinlambda * cosphi;
  theJacobian(4,3) = -p * coslambda * sinphi;
  theJacobian(5,1) = -q * p2 * coslambda * sinphi;
  theJacobian(5,2) = -p * sinlambda * sinphi;
  theJacobian(5,3) = p * coslambda * cosphi;
  theJacobian(6,1) = -q * p2 * sinlambda;
  theJacobian(6,2) = p * coslambda;
  theJacobian(6,3) = 0.;

  //ErrorPropagation: 
  //    C(Cart) = R(6*6) * J(6*5) * C(Curvi) * J(5*6)_T * R(6*6)_T
  theJacobian = R*theJacobian;
}

const AlgebraicMatrix& JacobianCurvilinearToCartesian::jacobian() const {
  return theJacobian;
}
