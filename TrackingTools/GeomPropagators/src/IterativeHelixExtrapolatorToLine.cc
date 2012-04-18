#include "TrackingTools/GeomPropagators/interface/IterativeHelixExtrapolatorToLine.h"
#include <iostream>

IterativeHelixExtrapolatorToLine::IterativeHelixExtrapolatorToLine
(const PositionType& point,
 const DirectionType& direction,
 const float curvature,
 const PropagationDirection propDir) :
  theX0(point.x()),
  theY0(point.y()),
  theZ0(point.z()),
  theRho(curvature),
  theQuadraticSolutionFromStart(point,direction,curvature,propDir),
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
template <class T> std::pair<bool,double>
IterativeHelixExtrapolatorToLine::genericPathLength (const T& object) const {
  //
  // Constants used for control of convergence
  //
  const int maxIterations(100);
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
  // Convergence criterion: maximal lateral displacement in a step < 1um
  //
  double maxDeltaS2 = 2*1.e-4/theSinTheta/theSinTheta/fabs(theRho);
  //
  bool first(true);
  while ( true ) {
    //
    // return empty solution vector if no convergence after maxIterations iterations
    //
    if ( --iteration<0 ) {
      return std::pair<bool,double>(false,0);
    }
    //
    // Use existing second order object at first pass, create temporary object
    // for subsequent passes.
    //
    std::pair<bool,double> deltaS1;
    if ( first ) {
      first = false;
      deltaS1 = theQuadraticSolutionFromStart.pathLength(object);
    }
    else {
      HelixExtrapolatorToLine2Order linearCrossing(xnew.x(),xnew.y(),xnew.z(),
						   pnew.x(),pnew.y(),
						   theCosTheta,theSinTheta,
						   theRho,anyDirection);
      deltaS1 = linearCrossing.pathLength(object);
    }
    if ( !deltaS1.first )  return deltaS1;
    //
    // Calculate total pathlength
    //
    dSTotal += deltaS1.second;
    PropagationDirection newDir = dSTotal>=0 ? alongMomentum : oppositeToMomentum;
    if ( propDir == anyDirection ) {
      propDir = newDir;
    }
    else {
      if ( newDir!=propDir )  return std::pair<bool,double>(false,0);
    }
    //
    // Check convergence
    //
    if ( deltaS1.second*deltaS1.second<maxDeltaS2 )  break;
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
HelixLineExtrapolation::PositionType
IterativeHelixExtrapolatorToLine::position (double s) const {
  // use result in double precision
  return PositionType(positionInDouble(s));
}

//
// Position on helix after a step of path length s in double precision.
//
HelixLineExtrapolation::PositionTypeDouble
IterativeHelixExtrapolatorToLine::positionInDouble (double s) const {
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
  //   1st order approximation.
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
    // Use 1st order.
    return theQuadraticSolutionFromStart.positionInDouble(theCachedS);
  }
}

//
// Direction vector on helix after a step of path length s.
//
HelixLineExtrapolation::DirectionType
IterativeHelixExtrapolatorToLine::direction (double s) const {
  // use result in double precision
//   DirectionTypeDouble dir = directionInDouble(s);
//   return DirectionType(dir.x(),dir.y(),dir.z());
  return DirectionType(directionInDouble(s));
}

//
// Direction vector on helix after a step of path length s in double precision.
//
HelixLineExtrapolation::DirectionTypeDouble
IterativeHelixExtrapolatorToLine::directionInDouble (double s) const {
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
    // 1st order
    return theQuadraticSolutionFromStart.directionInDouble(theCachedS);
  }
}
