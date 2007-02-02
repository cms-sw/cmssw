#ifndef TRACKINGTOOLS_TRANSIENTRACKBUILDER_H
#define TRACKINGTOOLS_TRANSIENTRACKBUILDER_H

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Handle.h"

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
