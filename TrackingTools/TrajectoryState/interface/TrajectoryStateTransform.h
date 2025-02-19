#ifndef TrajectoryStateTransform_H
#define TrajectoryStateTransform_H

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class TrackingGeometry;
class Surface;
class MagneticField;

namespace trajectoryStateTransform {

  PTrajectoryStateOnDet persistentState( const TrajectoryStateOnSurface& ts,
					  unsigned int detid);

  TrajectoryStateOnSurface transientState( const PTrajectoryStateOnDet& ts,
					   const Surface* surface,
					   const MagneticField* field);

  /// Construct a FreeTrajectoryState from the reco::Track innermost or outermost state, 
  /// does not require access to tracking geometry
  FreeTrajectoryState initialFreeState( const reco::Track& tk,
					const MagneticField* field);

  FreeTrajectoryState innerFreeState( const reco::Track& tk,
				      const MagneticField* field);
  FreeTrajectoryState outerFreeState( const reco::Track& tk,
				      const MagneticField* field);

  /// Construct a TrajectoryStateOnSurface from the reco::Track innermost or outermost state, 
  /// requires access to tracking geometry
  TrajectoryStateOnSurface innerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field);
  TrajectoryStateOnSurface outerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field);

}

// backward compatibility
struct TrajectoryStateTransform {};


#endif
