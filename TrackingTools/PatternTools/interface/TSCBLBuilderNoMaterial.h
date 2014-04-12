#ifndef TSCBLBuilderNoMaterial_H
#define TSCBLBuilderNoMaterial_H

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"

/**
 * This class builds a TrajectoryStateClosestToBeamLine given an original 
 * FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the beam line.
 * It is to be used when there is no material between the state and the BeamLine
 */

class TSCBLBuilderNoMaterial : public TrajectoryStateClosestToBeamLineBuilder
{
public: 

  virtual TrajectoryStateClosestToBeamLine operator()
    (const FTS& originalFTS, const reco::BeamSpot & beamSpot) const;

  virtual ~TSCBLBuilderNoMaterial() {};

};
#endif
