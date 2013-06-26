#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"

using namespace reco;
using namespace std;

TransientTrack TransientTrackFromFTSFactory::build (const FreeTrajectoryState & fts) const {
  return TransientTrack(new TransientTrackFromFTS(fts));
}

TransientTrack TransientTrackFromFTSFactory::build (const FreeTrajectoryState & fts,
	const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry) const {
  return TransientTrack(new TransientTrackFromFTS(fts, trackingGeometry));
}

