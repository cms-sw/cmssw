#ifndef StraightLinePlaneCrossing_H_
#define StraightLinePlaneCrossing_H_

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <utility>

class Plane;

/** Calculates intersections of a line with a plane.
 */

class StraightLinePlaneCrossing {
public:
  /** The types for position and direction are frame-neutral
   *  (not global, local, etc.) so this interface can be used
   *  in any frame. Of course, the helix and the plane must be defined 
   *  in the same frame, which is also the frame of the result.
   */
  typedef Basic3DVector<float>   PositionType;
  typedef Basic3DVector<float>   DirectionType;

public:
  /** Constructor using point and momentum.
   */
  StraightLinePlaneCrossing(const PositionType& point,
			    const DirectionType& momentum,
			    const PropagationDirection propDir = alongMomentum);
  // destructor
  ~StraightLinePlaneCrossing() {}

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane.
   */
  std::pair<bool,double> pathLength (const Plane& plane) const;

  /** Position at pathlength s from the starting point.
   */
  PositionType position (double s) const {return PositionType(theX0+s*theP0);}

  /** Simplified interface in case the path length is not needed
   */
  std::pair<bool,PositionType> position(const Plane& plane) const;

private:
  //
  // single precision vectors sufficient for internal use
  //
//   typedef Basic3DVector<double>  PositionTypeDouble;
//   typedef Basic3DVector<double>  DirectionTypeDouble;
  typedef Basic3DVector<float>  PositionTypeDouble;
  typedef Basic3DVector<float>  DirectionTypeDouble;

  const PositionTypeDouble theX0;
  const PositionTypeDouble theP0;
  const PropagationDirection thePropDir;
};

#endif
