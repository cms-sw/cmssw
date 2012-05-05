#ifndef HelixPlaneCrossing_H
#define HelixPlaneCrossing_H

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include <utility>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class Plane;

/** Abstract interface for the crossing of a helix with a plane.
 */

class HelixPlaneCrossing {
public:
  /// the helix is passed to the constructor and does not appear in the interface

  /** The types for position and direction are frame-neutral
   *  (not global, local, etc.) so this interface can be used
   *  in any frame. Of course, the helix and the plane must be defined 
   *  in the same frame, which is also the frame of the result.
   */
  typedef Basic3DVector<float>   PositionType;
  typedef Basic3DVector<float>   DirectionType;

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the plane. The 
   *  starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength( const Plane&) = 0;

  /** Returns the position along the helix that corresponds to path
   *  length "s" from the starting point. If s is obtained from the
   *  pathLength method the position is the destination point, i.e.
   *  the position of the crossing with a plane (if it exists!) 
   *  is given by position( pathLength( plane)).
   */
  virtual PositionType position( double s) const = 0;

  /** Returns the direction along the helix that corresponds to path
   *  length "s" from the starting point. As for position,
   *  the direction of the crossing with a plane (if it exists!) 
   *  is given by direction( pathLength( plane)).
   */
  virtual DirectionType direction( double s) const = 0;

};

#endif
