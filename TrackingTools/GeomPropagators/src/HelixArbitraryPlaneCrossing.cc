#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <cmath>
#include <iostream>

HelixArbitraryPlaneCrossing::HelixArbitraryPlaneCrossing(const PositionType& point,
							 const DirectionType& direction,
							 const float curvature,
							 const PropagationDirection propDir) :
  theX0(point.x()),
  theY0(point.y()),
  theZ0(point.z()),
  theRho(curvature),
  theQuadraticCrossingFromStart(point,direction,curvature,propDir),
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
  double pt = px*px+py*py;
  double p = sqrt(pt+pz*pz);
  pt = sqrt(pt);
  theCosPhi0 = px/pt;
  theSinPhi0 = py/pt;
  theCosTheta = pz/p;
  theSinTheta = pt/p;
}
//
// Propagation status and path length to intersection
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing::pathLength(const Plane& plane) {
  //
  // Constants used for control of convergence
  //
  const int maxIterations(100);
  //
  // maximum distance to plane (taking into account numerical precision)
  //
  float maxNumDz = theNumericalPrecision*plane.position().mag();
  float safeMaxDist = (theMaxDistToPlane>maxNumDz?theMaxDistToPlane:maxNumDz);
  //
  // Prepare internal value of the propagation direction and position / direction vectors for iteration 
  //
  PropagationDirection propDir = thePropDir;
  PositionTypeDouble xnew(theX0,theY0,theZ0);
  DirectionTypeDouble pnew(theCosPhi0,theSinPhi0,theCosTheta/theSinTheta);
  //
  // Prepare iterations: count and total pathlength
  //
  int iteration(maxIterations);
  double dSTotal(0.);
  //
  bool first(true);
  while ( notAtSurface(plane,xnew,safeMaxDist) ) {
    //
    // return empty solution vector if no convergence after maxIterations iterations
    //
    if ( --iteration<0 ) {
      return std::pair<bool,double>(false,0);
    }
    //
    // Use existing 2nd order object at first pass, create temporary object
    // for subsequent passes.
    //
    std::pair<bool,double> deltaS2;
    if ( first ) {
      first = false;
      deltaS2 = theQuadraticCrossingFromStart.pathLength(plane);
    }
    else {
      HelixArbitraryPlaneCrossing2Order quadraticCrossing(xnew.x(),xnew.y(),xnew.z(),
							  pnew.x(),pnew.y(),
							  theCosTheta,theSinTheta,
							  theRho,
							  anyDirection);
      deltaS2 = quadraticCrossing.pathLength(plane);
    }
    if ( !deltaS2.first )  return deltaS2;
    //
    // Calculate and sort total pathlength (max. 2 solutions)
    //
    dSTotal += deltaS2.second;
    PropagationDirection newDir = dSTotal>=0 ? alongMomentum : oppositeToMomentum;
    if ( propDir == anyDirection ) {
      propDir = newDir;
    }
    else {
      if ( newDir!=propDir )  return std::pair<bool,double>(false,0);
    }
    //
    // Step forward by dSTotal.
    //
    xnew = positionInDouble(dSTotal);
    pnew = directionInDouble(dSTotal);
  }
  //
  // Return result
  //
  return std::pair<bool,double>(true,dSTotal);
}
//
// Position on helix after a step of path length s.
//
HelixPlaneCrossing::PositionType
HelixArbitraryPlaneCrossing::position (double s) const {
  // use result in double precision
  PositionTypeDouble pos = positionInDouble(s);
  return PositionType(pos.x(),pos.y(),pos.z());
}
//
// Position on helix after a step of path length s in double precision.
//
HelixArbitraryPlaneCrossing::PositionTypeDouble
HelixArbitraryPlaneCrossing::positionInDouble (double s) const {
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
//    if ( fabs(theCachedDPhi)>1.e-1 ) {
  if ( fabs(theCachedDPhi)>1.e-4 ) {
    // "standard" helix formula
    return PositionTypeDouble(theX0+(-theSinPhi0*(1.-theCachedCDPhi)+theCosPhi0*theCachedSDPhi)/theRho,
			      theY0+(theCosPhi0*(1.-theCachedCDPhi)+theSinPhi0*theCachedSDPhi)/theRho,
			      theZ0+theCachedS*theCosTheta);
    }
//    else if ( fabs(theCachedDPhi)>theNumericalPrecision ) {
//      // full helix formula, but avoiding (1-cos(deltaPhi)) for small angles
//      return PositionTypeDouble(theX0+(-theSinPhi0*theCachedSDPhi*theCachedSDPhi/(1.+theCachedCDPhi)+
//  				     theCosPhi0*theCachedSDPhi)/theRho,
//  			      theY0+(theCosPhi0*theCachedSDPhi*theCachedSDPhi/(1.+theCachedCDPhi)+
//  				     theSinPhi0*theCachedSDPhi)/theRho,
//  			      theZ0+theCachedS*theCosTheta);
//    }
  else {
    // Use 2nd order.
    return theQuadraticCrossingFromStart.positionInDouble(theCachedS);
  }
}
//
// Direction vector on helix after a step of path length s.
//
HelixPlaneCrossing::DirectionType
HelixArbitraryPlaneCrossing::direction (double s) const {
  // use result in double precision
  DirectionTypeDouble dir = directionInDouble(s);
  return DirectionType(dir.x(),dir.y(),dir.z());
}
//
// Direction vector on helix after a step of path length s in double precision.
//
HelixArbitraryPlaneCrossing::DirectionTypeDouble
HelixArbitraryPlaneCrossing::directionInDouble (double s) const {
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
    return DirectionTypeDouble(theCosPhi0*theCachedCDPhi-theSinPhi0*theCachedSDPhi,
			       theSinPhi0*theCachedCDPhi+theCosPhi0*theCachedSDPhi,
			       theCosTheta/theSinTheta);
  }
  else {
    // 2nd order
    return theQuadraticCrossingFromStart.directionInDouble(theCachedS);
  }
}
//   Iteration control: continue if distance to plane > theMaxDistToPlane. Includes 
//   protection for numerical precision (Surfaces work with single precision).
bool HelixArbitraryPlaneCrossing::notAtSurface (const Plane& plane,  				       
						const PositionTypeDouble& point,
						const float maxDist) const {
  float dz = plane.localZ(Surface::GlobalPoint(point.x(),point.y(),point.z()));
  return fabs(dz)>maxDist;
}

const float HelixArbitraryPlaneCrossing::theNumericalPrecision = 5.e-7;
const float HelixArbitraryPlaneCrossing::theMaxDistToPlane = 1.e-4;
