#include "TrackingTools/GeomPropagators/interface/StraightLineBarrelCylinderCrossing.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "TrackingTools/GeomPropagators/src/RealQuadEquation.h"

#include <cmath>

StraightLineBarrelCylinderCrossing::StraightLineBarrelCylinderCrossing 
(const GlobalPoint& startingPos, const GlobalVector& startingDir,
 const PropagationDirection propDir) :
  theX0(startingPos),
  theP0(startingDir.unit()),
  thePropDir(propDir) {}

std::pair<bool,double>
StraightLineBarrelCylinderCrossing::pathLength (const Cylinder& cylinder) const
{
  //
  // radius of cylinder and transversal position relative to axis
  //
  double R(cylinder.radius());
  GlobalPoint axis(cylinder.toGlobal(Cylinder::LocalPoint(0.,0.)));
  PositionType2D xt2d(theX0.x()-axis.x(),theX0.y()-axis.y());
  //
  // transverse direction
  // 
  DirectionType2D pt2d(theP0.x(),theP0.y());
  //
  // solution of quadratic equation for s - assume |theP0|=1
  //
  RealQuadEquation eq(pt2d.mag2(),2.*xt2d.dot(pt2d),xt2d.mag2()-R*R);
  if ( !eq.hasSolution )  return std::pair<bool,double>(false,0.);
  //
  // choice of solution and verification of direction
  //
  return chooseSolution(eq.first,eq.second);
}

std::pair<bool,double>
StraightLineBarrelCylinderCrossing::chooseSolution (const double s1, 
						    const double s2) const
{
  //
  // follows the logic implemented in HelixBarrelCylinderCrossing
  //
  if ( thePropDir==anyDirection ) {
    return std::pair<bool,double>(true,(fabs(s1)<fabs(s2)?s1:s2));
  }
  else {
    int propSign = thePropDir==alongMomentum ? 1 : -1;
    if ( s1*s2 < 0) {
      // if different signs return the positive one
      return std::pair<bool,double>(true,((s1*propSign>0)?s1:s2));
    }
    else if ( s1*propSign>0 ) {
      // if both positive, return the shortest
      return std::pair<bool,double>(true,(fabs(s1)<fabs(s2)?s1:s2));
    }
  }
  return std::pair<bool,double>(false,0.);
}
