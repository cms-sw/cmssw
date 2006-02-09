#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

JacobianCartesianToCurvilinear::
JacobianCartesianToCurvilinear(const GlobalTrajectoryParameters& globalParameters) : theJacobian(5, 6, 0) {
  
  GlobalVector xt = globalParameters.momentum();
  //GlobalVector yt(xt.y(), -xt.x(), 0.); \\wrong direction of the axis
  //GlobalVector zt(xt.x()*xt.z(), xt.y()*xt.z(), -xt.perp2()); \\and then also on this one
  GlobalVector yt(-xt.y(), xt.x(), 0.);
  GlobalVector zt = xt.cross(yt);
  GlobalVector pvec = globalParameters.momentum();
  double pt = pvec.perp(), p = pvec.mag();
  double px = pvec.x(), py = pvec.y(), pz = pvec.z();
  double pt2 = pt*pt, p2 = p*p, p3 = p*p*p;
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
  R(1,1) = xt.x(); R(1,2) = xt.y(); R(1,3) = xt.z();
  R(2,1) = yt.x(); R(2,2) = yt.y(); R(2,3) = yt.z();
  R(3,1) = zt.x(); R(3,2) = zt.y(); R(3,3) = zt.z();
  R(4,4) = 1.;
  R(5,5) = 1.;
  R(6,6) = 1.;

  theJacobian(1,4) = -q*px/p3;        theJacobian(1,5) = -q*py/p3;        theJacobian(1,6) = -q*pz/p3;
  if(fabs(pt) > 0){
    //theJacobian(2,4) = (px*pz)/(pt*p2); theJacobian(2,5) = (py*pz)/(pt*p2); theJacobian(2,6) = -pt/p2; //wrong sign
    theJacobian(2,4) = -(px*pz)/(pt*p2); theJacobian(2,5) = -(py*pz)/(pt*p2); theJacobian(2,6) = pt/p2;
    theJacobian(3,4) = -py/pt2;         theJacobian(3,5) = px/pt2;          theJacobian(3,6) = 0.;
  }
  theJacobian(4,2) = 1.;
                 theJacobian(5,3) = 1.;
  theJacobian = theJacobian * R;
}

const AlgebraicMatrix& JacobianCartesianToCurvilinear::jacobian() const{
  return theJacobian;
}
