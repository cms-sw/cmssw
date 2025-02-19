#ifndef TransientTrackFromFTSFactory_H
#define TransientTrackFromFTSFactory_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

  /**
   * Helper class to build TransientTrack from a FreeTrajectoryState.
  */

class TransientTrackFromFTSFactory {
 public:

    reco::TransientTrack build (const FreeTrajectoryState & fts) const;
    reco::TransientTrack build (const FreeTrajectoryState & fts,
	const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry)  const;

};


#endif
