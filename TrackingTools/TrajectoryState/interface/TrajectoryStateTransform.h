#ifndef TrajectoryStateTransform_H
#define TrajectoryStateTransform_H

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackReco/interface/Track.h"

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class TrackingGeometry;
class Surface;
class MagneticField;

class TrajectoryStateTransform {
public:

  PTrajectoryStateOnDet* persistentState( const TrajectoryStateOnSurface& ts,
					  unsigned int detid) const;

  TrajectoryStateOnSurface transientState( const PTrajectoryStateOnDet& ts,
					   const Surface* surface,
					   const MagneticField* field) const;

  /// Construct a FreeTrajectoryState from the reco::Track innermost or outermost state, 
  /// does not require access to tracking geometry
  FreeTrajectoryState initialFreeState( const reco::Track& tk,
					const MagneticField* field) const;

  FreeTrajectoryState innerFreeState( const reco::Track& tk,
				      const MagneticField* field) const;
  FreeTrajectoryState outerFreeState( const reco::Track& tk,
				      const MagneticField* field) const;

  /// Construct a TrajectoryStateOnSurface from the reco::Track innermost or outermost state, 
  /// requires access to tracking geometry
  TrajectoryStateOnSurface innerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field) const;
  TrajectoryStateOnSurface outerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field) const;

};

#endif
