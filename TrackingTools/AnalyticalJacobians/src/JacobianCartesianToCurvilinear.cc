#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

JacobianCartesianToCurvilinear::
JacobianCartesianToCurvilinear(const GlobalTrajectoryParameters& globalParameters) : theJacobian() {
  
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
  
  AlgebraicMatrix66 R;
  R(0,0) = xt.x(); R(0,1) = xt.y(); R(0,2) = xt.z();
  R(1,0) = yt.x(); R(1,1) = yt.y(); R(1,2) = yt.z();
  R(2,0) = zt.x(); R(2,1) = zt.y(); R(2,2) = zt.z();
  R(3,3) = 1.;
  R(4,4) = 1.;
  R(5,5) = 1.;

  theJacobian(0,3) = -q*px/p3;        theJacobian(0,4) = -q*py/p3;        theJacobian(0,5) = -q*pz/p3;
  if(fabs(pt) > 0){
    //theJacobian(1,3) = (px*pz)/(pt*p2); theJacobian(1,4) = (py*pz)/(pt*p2); theJacobian(1,5) = -pt/p2; //wrong sign
    theJacobian(1,3) = -(px*pz)/(pt*p2); theJacobian(1,4) = -(py*pz)/(pt*p2); theJacobian(1,5) = pt/p2;
    theJacobian(2,3) = -py/pt2;         theJacobian(2,4) = px/pt2;          theJacobian(2,5) = 0.;
  }
  theJacobian(3,1) = 1.;
  theJacobian(4,2) = 1.;
  theJacobian = theJacobian * R;
  //dbg::dbg_trace(1,"Ca2Cu", globalParameters.vector(),theJacobian);
}
const AlgebraicMatrix JacobianCartesianToCurvilinear::jacobian_old() const{
  return asHepMatrix(theJacobian);
}
const AlgebraicMatrix56& JacobianCartesianToCurvilinear::jacobian() const{
  return theJacobian;
}
