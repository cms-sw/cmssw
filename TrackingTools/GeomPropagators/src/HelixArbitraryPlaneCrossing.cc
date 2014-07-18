#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <vdt/vdtMath.h>
#include <iostream>
#include "FWCore/Utilities/interface/Likely.h"

#ifdef VI_DEBUG
struct MaxIter {
   ~MaxIter() { std::cout << "maxiter " << v << std::endl; }
    void operator()(int i) { v= std::min(v,i);}    
  int v=100;
};
#else
struct MaxIter { void operator()(int)const{}};
#endif
static MaxIter maxiter;


HelixArbitraryPlaneCrossing::HelixArbitraryPlaneCrossing(const PositionType& point,
							 const DirectionType& direction,
							 const float curvature,
							 const PropagationDirection propDir) :
  theQuadraticCrossingFromStart(point,direction,curvature,propDir),
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
// Propagation status and path length to intersection
//
std::pair<bool,double>
HelixArbitraryPlaneCrossing::pathLength(const Plane& plane) {
  //
  // Constants used for control of convergence
  //
  constexpr int maxIterations = 20;
  //
  // maximum distance to plane (taking into account numerical precision)
  //
  float maxNumDz = theNumericalPrecision*plane.position().mag();
  float safeMaxDist = (theMaxDistToPlane>maxNumDz?theMaxDistToPlane:maxNumDz);
  //
  // Prepare internal value of the propagation direction and position / direction vectors for iteration 
  //
  
  float dz = plane.localZ(Surface::GlobalPoint(theX0,theY0,theZ0));
  if (std::abs(dz)<safeMaxDist) return std::make_pair(true,0.);

  bool notFail;
  double dSTotal;
  // Use existing 2nd order object at first pass
  std::tie(notFail,dSTotal) = theQuadraticCrossingFromStart.pathLength(plane);
  if unlikely(!notFail) return std::make_pair(notFail,dSTotal);
  auto  xnew = positionInDouble(dSTotal);

  auto propDir = thePropDir;
  auto newDir = dSTotal>=0 ? alongMomentum : oppositeToMomentum;
  if ( propDir == anyDirection ) {
      propDir = newDir;
  }  else {
     if unlikely( newDir!=propDir )  return std::pair<bool,double>(false,0);
  }
 

  //
  // Prepare iterations: count and total pathlength
  //
  auto iteration = maxIterations;
  while ( notAtSurface(plane,xnew,safeMaxDist) ) {
    //
    // return empty solution vector if no convergence after maxIterations iterations
    //
    if unlikely( --iteration == 0 ) {
      LogDebug("HelixArbitraryPlaneCrossing") << "pathLength : no convergence";
      return std::pair<bool,double>(false,0);
    }

    //
    // create temporary object for subsequent passes.
    auto  pnew = directionInDouble(dSTotal);
    HelixArbitraryPlaneCrossing2Order quadraticCrossing(xnew.x(),xnew.y(),xnew.z(),
							  pnew.x(),pnew.y(),
							  theCosTheta,theSinTheta,
							  theRho,
							  anyDirection);
      
    auto  deltaS2 = quadraticCrossing.pathLength(plane);
   
     
    if unlikely( !deltaS2.first )  return deltaS2;
    //
    // Calculate and sort total pathlength (max. 2 solutions)
    //
    dSTotal += deltaS2.second;
    auto newDir = dSTotal>=0 ? alongMomentum : oppositeToMomentum;
    if ( propDir == anyDirection ) {
      propDir = newDir;
    }
    else {
      if unlikely( newDir!=propDir )  return std::pair<bool,double>(false,0);
    }
    //
    // Step forward by dSTotal.
    //
    xnew = positionInDouble(dSTotal);
  }
  //
  // Return result
  //
  maxiter(iteration);

  return std::make_pair(true,dSTotal);
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
  if unlikely( s!=theCachedS ) {
    theCachedS = s;
    theCachedDPhi = theCachedS*theRho*theSinTheta;
    vdt::fast_sincos(theCachedDPhi,theCachedSDPhi,theCachedCDPhi);
  }
  //
  // Calculate with appropriate formulation of full helix formula or with 
  //   2nd order approximation.
  //
//    if ( fabs(theCachedDPhi)>1.e-1 ) {
  if ( std::abs(theCachedDPhi)>1.e-4 ) {
    // "standard" helix formula
    double o = 1./theRho;
    return PositionTypeDouble(theX0+(-theSinPhi0*(1.-theCachedCDPhi)+theCosPhi0*theCachedSDPhi)*o,
			      theY0+( theCosPhi0*(1.-theCachedCDPhi)+theSinPhi0*theCachedSDPhi)*o,
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
  if unlikely( s!=theCachedS ) {  // very very unlikely!
    theCachedS = s;
    theCachedDPhi = theCachedS*theRho*theSinTheta;
    vdt::fast_sincos(theCachedDPhi,theCachedSDPhi,theCachedCDPhi);
  }

  if ( std::abs(theCachedDPhi)>1.e-4 ) {
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
  return std::abs(dz)>maxDist;
}

const float HelixArbitraryPlaneCrossing::theNumericalPrecision = 5.e-7f;
const float HelixArbitraryPlaneCrossing::theMaxDistToPlane = 1.e-4f;
