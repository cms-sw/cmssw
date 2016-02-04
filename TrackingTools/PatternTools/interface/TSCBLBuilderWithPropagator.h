#ifndef TSCBLBuilderWithPropagator_H
#define TSCBLBuilderWithPropagator_H

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "MagneticField/Engine/interface/MagneticField.h"

/**
 * This class builds a TrajectoryStateClosestToBeamLine given an original 
 * FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the beam line.
 * It is to be used when a specific propagator has to be used.
 */

class TSCBLBuilderWithPropagator : public TrajectoryStateClosestToBeamLineBuilder
{
public: 

  /// constructor with default geometrical propagator
  TSCBLBuilderWithPropagator(const MagneticField* field);
  
  /// constructor with user-supplied propagator
  TSCBLBuilderWithPropagator(const Propagator& u);

  virtual ~TSCBLBuilderWithPropagator(){};

  virtual TrajectoryStateClosestToBeamLine operator()
    (const FTS& originalFTS, const reco::BeamSpot & beamSpot) const;

private:
  DeepCopyPointerByClone<Propagator> thePropagator;

};
#endif
