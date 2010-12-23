#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

bool ClosestApproachInRPhi::calculate(const TrajectoryStateOnSurface & sta, 
				      const TrajectoryStateOnSurface & stb) 
{
  TrackCharge chargeA = sta.charge(); TrackCharge chargeB = stb.charge();
  GlobalVector momentumA = sta.globalMomentum();
  GlobalVector momentumB = stb.globalMomentum();
  GlobalPoint positionA = sta.globalPosition();
  GlobalPoint positionB = stb.globalPosition();
  paramA = sta.globalParameters();
  paramB = stb.globalParameters();
  // compute magnetic field ONCE 
  bz = sta.freeState()->parameters().magneticField().inTesla(positionA).z() * 2.99792458e-3;

  return compute(chargeA, momentumA, positionA, chargeB, momentumB, positionB);

}


bool ClosestApproachInRPhi::calculate(const FreeTrajectoryState & sta, 
				      const FreeTrajectoryState & stb)
{
  TrackCharge chargeA = sta.charge(); TrackCharge chargeB = stb.charge();
  GlobalVector momentumA = sta.momentum();
  GlobalVector momentumB = stb.momentum();
  GlobalPoint positionA = sta.position();
  GlobalPoint positionB = stb.position();
  paramA = sta.parameters();
  paramB = stb.parameters();
  // compute magnetic field ONCE 
  bz = sta.parameters().magneticField().inTesla(positionA).z() * 2.99792458e-3;

  return compute(chargeA, momentumA, positionA, chargeB, momentumB, positionB);

}

pair<GlobalPoint, GlobalPoint> ClosestApproachInRPhi::points() const
{
  if (!status_)
    throw cms::Exception("TrackingTools/PatternTools","ClosestApproachInRPhi::could not compute track crossing. Check status before calling this method!");
  return  pair<GlobalPoint, GlobalPoint> (posA, posB);
}


GlobalPoint 
ClosestApproachInRPhi::crossingPoint() const
{
  if (!status_)
    throw cms::Exception("TrackingTools/PatternTools","ClosestApproachInRPhi::could not compute track crossing. Check status before calling this method!");
  return  GlobalPoint(0.5*(posA.basicVector() + posB.basicVector()));
		     
}


float ClosestApproachInRPhi::distance() const
{
  if (!status_)
    throw cms::Exception("TrackingTools/PatternTools","ClosestApproachInRPhi::could not compute track crossing. Check status before calling this method!");
  return (posB - posA).mag();
}


bool ClosestApproachInRPhi::compute(const TrackCharge & chargeA, 
				    const GlobalVector & momentumA, 
				    const GlobalPoint & positionA, 
				    const TrackCharge & chargeB, 
				    const GlobalVector & momentumB, 
				    const GlobalPoint & positionB) 
{


  // centres and radii of track circles
  double xca, yca, ra;
  circleParameters(chargeA, momentumA, positionA, xca, yca, ra, bz);
  double xcb, ycb, rb;
  circleParameters(chargeB, momentumB, positionB, xcb, ycb, rb, bz);

  // points of closest approach in transverse plane
  double xg1, yg1, xg2, yg2;
  int flag = transverseCoord(xca, yca, ra, xcb, ycb, rb, xg1, yg1, xg2, yg2);
  if (flag == 0) {
    status_ = false;
    return false;
  }

  double xga, yga, zga, xgb, ygb, zgb;

  if (flag == 1) {
    // two crossing points on each track in transverse plane
    // select point for which z-coordinates on the 2 tracks are the closest
    double za1 = zCoord(momentumA, positionA, ra, xca, yca, xg1, yg1);
    double zb1 = zCoord(momentumB, positionB, rb, xcb, ycb, xg1, yg1);
    double za2 = zCoord(momentumA, positionA, ra, xca, yca, xg2, yg2);
    double zb2 = zCoord(momentumB, positionB, rb, xcb, ycb, xg2, yg2);

    if (abs(zb1 - za1) < abs(zb2 - za2)) {
      xga = xg1; yga = yg1; zga = za1; zgb = zb1;
    }
    else {
      xga = xg2; yga = yg2; zga = za2; zgb = zb2;
    }
    xgb = xga; ygb = yga;
  }
  else {
    // one point of closest approach on each track in transverse plane
    xga = xg1; yga = yg1;
    zga = zCoord(momentumA, positionA, ra, xca, yca, xga, yga);
    xgb = xg2; ygb = yg2;
    zgb = zCoord(momentumB, positionB, rb, xcb, ycb, xgb, ygb);
  }

  posA = GlobalPoint(xga, yga, zga);
  posB = GlobalPoint(xgb, ygb, zgb);
  status_ = true;
  return true;
}

pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters>
ClosestApproachInRPhi::trajectoryParameters () const
{
  if (!status_)
    throw cms::Exception("TrackingTools/PatternTools","ClosestApproachInRPhi::could not compute track crossing. Check status before calling this method!");
  pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters> 
    ret ( newTrajectory( posA, paramA, bz),
          newTrajectory( posB, paramB, bz) );
  return ret;
}

GlobalTrajectoryParameters 
ClosestApproachInRPhi::newTrajectory( const GlobalPoint & newpt, const GlobalTrajectoryParameters & oldgtp, double bz )
{
  // First we need the centers of the circles.
  double qob = oldgtp.charge()/bz;
  double xc =  oldgtp.position().x() + qob *  oldgtp.momentum().y();
  double yc =  oldgtp.position().y() - qob *  oldgtp.momentum().x();
  
  // now we do a translation, move the center of circle to (0,0,0).
  double dx1 = oldgtp.position().x() - xc;
  double dy1 = oldgtp.position().y() - yc;
  double dx2 = newpt.x() - xc;
  double dy2 = newpt.y() - yc;
  // and of course....
  double npx = (newpt.y()-yc)/qob;
  double npy = (xc-newpt.x())/qob;
  
  // now for the angles:
  double cosphi = ( dx1 * dx2 + dy1 * dy2 ) / 
    ( sqrt ( dx1 * dx1 + dy1 * dy1 ) * sqrt ( dx2 * dx2 + dy2 * dy2 ));
  double sinphi = - oldgtp.charge() * sqrt ( 1 - cosphi * cosphi );
  
  // Finally, the new momenta:
  double px = cosphi * oldgtp.momentum().x() - sinphi * oldgtp.momentum().y();
  double py = sinphi * oldgtp.momentum().x() + cosphi * oldgtp.momentum().y();
  
  std::cout << px-npx << " " << py-npy << std::endl;

  GlobalVector vta ( npx, npy, oldgtp.momentum().z() );
  GlobalTrajectoryParameters gta( newpt , vta , oldgtp.charge(), &(oldgtp.magneticField()) );
  return gta;
}

void 
ClosestApproachInRPhi::circleParameters(const TrackCharge& charge, 
					const GlobalVector& momentum, 
					const GlobalPoint& position, 
					double& xc, double& yc, double& r,
					double bz)
{

  // compute radius of circle
  /** temporary code, to be replaced by call to curvature() when bug 
   *  is fixed. 
   */
//   double bz = MagneticField::inInverseGeV(position).z();

  // signed_r directed towards circle center, along F_Lorentz = q*v X B
  double qob = charge/bz;
  double signed_r = qob*momentum.transverse();
  r = abs(signed_r);
  /** end of temporary code
   */

  // compute centre of circle
  // double phi = momentum.phi();
  // xc = signed_r*sin(phi) + position.x();
  // yc = -signed_r*cos(phi) + position.y();
  xc =  position.x() + qob * momentum.y();
  yc =  position.y() - qob * momentum.x();

}


int 
ClosestApproachInRPhi::transverseCoord(double cxa, double cya, double ra, 
				       double cxb, double cyb, double rb, 
				       double & xg1, double & yg1, 
				       double & xg2, double & yg2)
{
  int flag = 0;
  double x1, y1, x2, y2;

  // new reference frame with origin in (cxa, cya) and x-axis 
  // directed from (cxa, cya) to (cxb, cyb)

  double d_ab = sqrt((cxb - cxa)*(cxb - cxa) + (cyb - cya)*(cyb - cya));
  if (d_ab == 0) { // concentric circles
    return 0;
  }
  // elements of rotation matrix
  double u = (cxb - cxa) / d_ab;
  double v = (cyb - cya) / d_ab;

  // conditions for circle intersection
  if (d_ab <= ra + rb && d_ab >= abs(rb - ra)) {

    // circles cross each other
    flag = 1;

    // triangle (ra, rb, d_ab)
    double cosphi = (ra*ra - rb*rb + d_ab*d_ab) / (2*ra*d_ab);
    double sinphi2 = 1. - cosphi*cosphi;
    if (sinphi2 < 0.) { sinphi2 = 0.; cosphi = 1.; }

    // intersection points in new frame
    double sinphi = sqrt(sinphi2);
    x1 = ra*cosphi; y1 = ra*sinphi; x2 = x1; y2 = -y1;
  } 
  else if (d_ab > ra + rb) {

    // circles are external to each other
    flag = 2;

    // points of closest approach in new frame 
    // are on line between 2 centers
    x1 = ra; y1 = 0; x2 = d_ab - rb; y2 = 0;
  }
  else if (d_ab < abs(rb - ra)) {

    // circles are inside each other
    flag = 2;

    // points of closest approach in new frame are on line between 2 centers
    // choose 2 closest points
    double sign = 1.;
    if (ra <= rb) sign = -1.;
    x1 = sign*ra; y1 = 0; x2 = d_ab + sign*rb; y2 = 0;
  }
  else {
    return 0;
  }

  // intersection points in global frame, transverse plane
  xg1 = u*x1 - v*y1 + cxa; yg1 = v*x1 + u*y1 + cya;
  xg2 = u*x2 - v*y2 + cxa; yg2 = v*x2 + u*y2 + cya;

  return flag;
}


double 
ClosestApproachInRPhi::zCoord(const GlobalVector& mom, 
			      const GlobalPoint& pos, 
			      double r, double xc, double yc, 
			      double xg, double yg)
{

  // starting point
  double x = pos.x(); double y = pos.y(); double z = pos.z();

  double px = mom.x(); double py = mom.y(); double pz = mom.z();

  // rotation angle phi from starting point to crossing point (absolute value)
  // -- compute sin(phi/2) if phi smaller than pi/4, 
  // -- cos(phi) if phi larger than pi/4
  double phi = 0.;
  double sinHalfPhi = sqrt((x-xg)*(x-xg) + (y-yg)*(y-yg))/(2*r);
  if (sinHalfPhi < 0.383) { // sin(pi/8)
    phi = 2*asin(sinHalfPhi);
  }
  else {
    double cosPhi = ((x-xc)*(xg-xc) + (y-yc)*(yg-yc))/(r*r);
    if (std::abs(cosPhi) > 1) cosPhi = (cosPhi > 0 ? 1 : -1);
    phi = abs(acos(cosPhi));
  }
  // -- sign of phi
  double signPhi = ((x - xc)*(yg - yc) - (xg - xc)*(y - yc) > 0) ? 1. : -1.;

  // sign of track angular momentum
  // if rotation is along angular momentum, delta z is along pz
  double signOmega = ((x - xc)*py - (y - yc)*px > 0) ? 1. : -1.;

  // delta z
  // -- |dz| = |cos(theta) * path along helix|
  //         = |cos(theta) * arc length along circle / sin(theta)|
  double dz = signPhi*signOmega*(pz/mom.transverse())*phi*r;

  return z + dz;
}
