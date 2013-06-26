#ifndef _COMMONRECO_PropagationDirectionChooser_H_
#define _COMMONRECO_PropagationDirectionChooser_H_

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

class FreeTrajectoryState;
class Surface;
class Cylinder;
class Plane;

/** Determination of propagation direction towards a surface.
 * Uses code from the old BidirectionalPropagator. */

class PropagationDirectionChooser {
public:

  PropagationDirectionChooser() {}
  
  ~PropagationDirectionChooser() {}

  PropagationDirection operator() (const FreeTrajectoryState&,
				   const Surface&) const;

  PropagationDirection operator() (const FreeTrajectoryState&,
				   const Plane&) const;

  PropagationDirection operator() (const FreeTrajectoryState&,
				   const Cylinder&) const;

};

#endif


