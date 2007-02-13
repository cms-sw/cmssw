#ifndef _COMMONDET_TRAJECTORY_EXTRAPOLATOR_TO_LINE_H_
#define _COMMONDET_TRAJECTORY_EXTRAPOLATOR_TO_LINE_H_


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Surface/interface/Line.h"

//class MagneticField;

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"


//#include "CommonDet/PatternPrimitives/interface/TrajectoryStateOnSurface.h"
//#include "CommonDet/DetGeometry/interface/Line.h"

class FreeTrajectoryState;
class Propagator;

/** Extrapolate to impact point with respect to a line, 
  * i.e. point of closest approach to a line in 3D.
  * The line is defined starting from a global point (position of a point
  * belonging to the line) and a global vector standing for the direction of 
  * the line.
  * The surface of the returned TrajectoryStateOnSurface 
  * is chosen perpendicular to momentum at the impact point.
  * The origin is at the point of the line closest to the helix, and
  * the axes of the local coordinate system (x_loc, y_loc, z_loc) are 
  * z_loc // trajectory direction at impact point;
  * x_loc along the minimum distance and pointing toward the helix
  * y_loc forms a right-handed system with the other axes.
  *
  * Any question please contact Gabriele.Segneri@cern.ch
  * Local co-ordinate system changed for compatibility with
  *   Vertex sub-system: wolfgang adam.
  */


class TrajectoryExtrapolatorToLine {

public:
  /// extrapolation with default (=geometrical) propagator
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState& fts,
				       const Line & L, const MagneticField* field) const;
  /// extrapolation with user-supplied propagator
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState& fts,
				       const Line& L,
				       const Propagator& p) const;


  // this is the old ORCA RecTrack::stateAtLine()
  TrajectoryStateOnSurface stateAtLine(const reco::TransientTrack& theTrack, const Line & aLine) const; 


};

#endif




