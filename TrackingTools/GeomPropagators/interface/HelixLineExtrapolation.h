#ifndef HelixLineExtrapolation_H
#define HelixLineExtrapolation_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <utility>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class Line;

/** Abstract interface for the extrapolation of a helix to
 *  the closest approach to a line.
 */

class HelixLineExtrapolation {
public:
  /** The types for position and direction are frame-neutral
   *  (not global, local, etc.) so this interface can be used
   *  in any frame. Of course, the helix and the plane must be defined 
   *  in the same frame, which is also the frame of the result.
   */
  typedef Basic3DVector<float>   PositionType;
  typedef Basic3DVector<float>   DirectionType;
  typedef Basic3DVector<double>  PositionTypeDouble;
  typedef Basic3DVector<double>  DirectionTypeDouble;

public:
  //
  // the helix is passed to the constructor and does not appear in the interface
  //

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach
   *  to the point. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const GlobalPoint& point) const = 0;

  /** Propagation status (true if valid) and (signed) path length 
   *  along the helix from the starting point to the closest approach
   *  to the line. The starting point is given in the constructor.
   */
  virtual std::pair<bool,double> pathLength (const Line& line) const = 0;

  /** Returns the position along the helix that corresponds to path
   *  length "s" from the starting point. If s is obtained from the
   *  pathLength method the position is the destination point, i.e.
   *  the position at the closest approach (if it exists!) 
   *  is given by position( pathLength(line) ).
   */
  virtual PositionType position (double s) const = 0;

  /** Returns the direction along the helix that corresponds to path
   *  length "s" from the starting point. As for position,
   *  the direction at the closest approach (if it exists!) 
   *  is given by direction( pathLength(line) ).
   */
  virtual DirectionType direction (double s) const = 0;

};

#endif
