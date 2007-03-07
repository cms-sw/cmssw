#ifndef StraightLineCylinderCrossing_H
#define StraightLineCylinderCrossing_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <utility>

class Cylinder;

/** Calculates the intersection of a straight line with a barrel cylinder.
 */ 

class StraightLineCylinderCrossing {

public:
  /** Constructor in local frame
   */
  StraightLineCylinderCrossing (const LocalPoint& startingPos,
				const LocalVector& startingDir,
				const PropagationDirection propDir=alongMomentum,
				double tolerance=0);

  /** Propagation status (true if valid) and (signed) path length 
   *  along the line from the starting point to the cylinder.
   */
  std::pair<bool,double> pathLength (const Cylinder& cyl) const;

  /** Returns the position along the line that corresponds to path
   *  length "s" from the starting point. If s is obtained from the
   *  pathLength method the position is the intersection point
   *  with the cylinder.
   */
  LocalPoint position (const double s) const { return LocalPoint(theX0+s*theP0);}

private:
  /// Chooses the right solution w.r.t. the propagation direction.
  std::pair<bool,double> chooseSolution (const double s1, const double s2) const;

private:
  //
  // single precision is sufficient for intermediate vectors
  //
  typedef LocalPoint   PositionType;
  typedef LocalVector  DirectionType;
  typedef Basic2DVector<float>  PositionType2D;
  typedef Basic2DVector<float>  DirectionType2D;

  const PositionType   theX0;
  const DirectionType  theP0;
  const PropagationDirection thePropDir;
  double theTolerance;
};

#endif
