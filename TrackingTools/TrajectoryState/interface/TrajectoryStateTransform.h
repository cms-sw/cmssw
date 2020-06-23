#ifndef TrajectoryStateTransform_H
#define TrajectoryStateTransform_H

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class TrackingGeometry;
class Surface;
class MagneticField;

namespace trajectoryStateTransform {

  PTrajectoryStateOnDet persistentState(const TrajectoryStateOnSurface& ts, unsigned int detid);

  TrajectoryStateOnSurface transientState(const PTrajectoryStateOnDet& ts,
                                          const Surface* surface,
                                          const MagneticField* field);

  /// Construct a FreeTrajectoryState from the reco::Track innermost or outermost state,
  /// does not require access to tracking geometry
  FreeTrajectoryState initialFreeState(const reco::Track& tk, const MagneticField* field, bool withErr = true);
  FreeTrajectoryState initialFreeStateTTrack(const TTTrack< Ref_Phase2TrackerDigi_ >& tk, const MagneticField* field, bool withErr = false);

  FreeTrajectoryState innerFreeState(const reco::Track& tk, const MagneticField* field, bool withErr = true);
  FreeTrajectoryState outerFreeState(const reco::Track& tk, const MagneticField* field, bool withErr = true);

  /// Construct a TrajectoryStateOnSurface from the reco::Track innermost or outermost state,
  /// requires access to tracking geometry
  TrajectoryStateOnSurface innerStateOnSurface(const reco::Track& tk,
                                               const TrackingGeometry& geom,
                                               const MagneticField* field,
                                               bool withErr = true);
  TrajectoryStateOnSurface outerStateOnSurface(const reco::Track& tk,
                                               const TrackingGeometry& geom,
                                               const MagneticField* field,
                                               bool withErr = true);

}  // namespace trajectoryStateTransform

// backward compatibility
struct TrajectoryStateTransform {};

#endif
