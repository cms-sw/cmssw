#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <cmath>
#include <cfloat>

HelixForwardPlaneCrossing::HelixForwardPlaneCrossing(const PositionType& point,
						     const DirectionType& direction,
						     const float curvature,
						     const PropagationDirection propDir) :
  theX0(point.x()),
  theY0(point.y()),
  theZ0(point.z()),
  theRho(curvature),
  thePropDir(propDir),
  theCachedS(0),
  theCachedDPhi(0.),
  theCachedSDPhi(0.),
  theCachedCDPhi(1.)
{
  //
  // Components of direction vector (with correct normalisation)
  //
  double px = direction.x();
  double py = direction.y();
  double pz = direction.z();
  double pt2 = px*px+py*py;
  double p2 = pt2+pz*pz;
  double pI = 1./sqrt(p2);
  double ptI = 1./sqrt(pt2);
  theCosPhi0 = px*ptI;
  theSinPhi0 = py*ptI;
  theCosTheta = pz*pI;
  theSinTheta = pt2*ptI*pI;

}
//
// Propagation status  and path length to intersection
//
std::pair<bool,double>
HelixForwardPlaneCrossing::pathLength(const Plane& plane) {
  //
  // Protect against p_z=0 and calculate path length
  //
  if ( std::abs(theCosTheta/theSinTheta)<FLT_MIN )  return std::pair<bool,double>(false,0);

  double dS = (plane.position().z()-theZ0) / theCosTheta;

  if ( (thePropDir==alongMomentum && dS<0.) ||
       (thePropDir==oppositeToMomentum && dS>0.) )  return std::pair<bool,double>(false,0);
  //
  // Return result
  //
  return std::pair<bool,double>(true,dS);
}
//
// Position on helix after a step of path length s. 
//
HelixPlaneCrossing::PositionType
HelixForwardPlaneCrossing::position (double s) const {
  //
  // Calculate delta phi (if not already available)
  //
  if ( s!=theCachedS ) {
    theCachedS = s;
    theCachedDPhi = theCachedS*theRho*theSinTheta;
    theCachedSDPhi = sin(theCachedDPhi);
    theCachedCDPhi = cos(theCachedDPhi);
  }
  //
  // Calculate with appropriate formulation of full helix formula or with 
  //   2nd order approximation.
  //
  if ( std::abs(theCachedDPhi)>1.e-4 ) {
    // "standard" helix formula
    // "standard" helix formula
    double or = 1./theRho;
    return PositionTypeDouble(theX0+(-theSinPhi0*(1.-theCachedCDPhi)+theCosPhi0*theCachedSDPhi)*or,
			      theY0+( theCosPhi0*(1.-theCachedCDPhi)+theSinPhi0*theCachedSDPhi)*or,
			      theZ0+theCachedS*theCosTheta);
  }
  else {
    // 2nd order
    double st = theCachedS*theSinTheta;
    return PositionType(theX0+(theCosPhi0-st*0.5*theRho*theSinPhi0)*st,
			theY0+(theSinPhi0+st*0.5*theRho*theCosPhi0)*st,
			theZ0+st*theCosTheta/theSinTheta);
  }
}
//
// Direction vector on helix after a step of path length s.
//
HelixPlaneCrossing::DirectionType
HelixForwardPlaneCrossing::direction (double s) const {
  //
  // Calculate delta phi (if not already available)
  //
  if ( s!=theCachedS ) {
    theCachedS = s;
    theCachedDPhi = theCachedS*theRho*theSinTheta;
    theCachedSDPhi = sin(theCachedDPhi);
    theCachedCDPhi = cos(theCachedDPhi);
  }

  if ( fabs(theCachedDPhi)>1.e-4 ) {
    // full helix formula
    return DirectionType(theCosPhi0*theCachedCDPhi-theSinPhi0*theCachedSDPhi,
			 theSinPhi0*theCachedCDPhi+theCosPhi0*theCachedSDPhi,
			 theCosTheta/theSinTheta);
  }
  else {
    // 2nd order
    double dph = theCachedS*theRho*theSinTheta;
    return DirectionType(theCosPhi0-(theSinPhi0+0.5*theCosPhi0*dph)*dph,
			 theSinPhi0+(theCosPhi0-0.5*theSinPhi0*dph)*dph,
			 theCosTheta/theSinTheta);
  }
}

const float HelixForwardPlaneCrossing::theNumericalPrecision = 5.e-7;
