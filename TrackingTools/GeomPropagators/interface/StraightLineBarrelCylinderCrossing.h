#ifndef StraightLineBarrelCylinderCrossing_H
#define StraightLineBarrelCylinderCrossing_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <utility>

class Cylinder;

/** Calculates the intersection of a straight line with a barrel cylinder.
 */ 

class StraightLineBarrelCylinderCrossing {

public:
  /** Constructor uses global frame (barrel cylinders are by
   *  definition in global co-ordinates)
   */
  StraightLineBarrelCylinderCrossing (const GlobalPoint& startingPos,
				      const GlobalVector& startingDir,
				      const PropagationDirection propDir);

  /** Propagation status (true if valid) and (signed) path length 
   *  along the line from the starting point to the cylinder.
   */
  std::pair<bool,double> pathLength (const Cylinder& cyl) const;

  /** Returns the position along the line that corresponds to path
   *  length "s" from the starting point. If s is obtained from the
   *  pathLength method the position is the intersection point
   *  with the cylinder.
   */
  GlobalPoint position (const double s) const { return GlobalPoint(theX0+s*theP0);}

private:
  /// Chooses the right solution w.r.t. the propagation direction.
  std::pair<bool,double> chooseSolution (const double s1, const double s2) const;

private:
  //
  // single precision is sufficient for intermediate vectors
  //
  typedef GlobalPoint   PositionType;
  typedef GlobalVector  DirectionType;
  typedef Basic2DVector<float>  PositionType2D;
  typedef Basic2DVector<float>  DirectionType2D;

  const PositionType   theX0;
  const DirectionType  theP0;
  const PropagationDirection thePropDir;
};

#endif
