#ifndef TrajectoryStateClosestToBeamLineBuilder_H
#define TrajectoryStateClosestToBeamLineBuilder_H

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

/**
 * This is the abstract class to build a TrajectoryStateClosestToBeamLine given an original 
 * FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the beam line.
 */

class TrajectoryStateClosestToBeamLineBuilder
{
public: 

  typedef FreeTrajectoryState		FTS;

  virtual TrajectoryStateClosestToBeamLine operator()
    (const FTS& originalFTS, const reco::BeamSpot & beamSpot) const = 0;

};
#endif
