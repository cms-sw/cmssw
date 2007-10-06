#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TrackProducer/plugins/TrackProducer.h"
#include "RecoTracker/TrackProducer/plugins/TrackRefitter.h"
#include "RecoTracker/TrackProducer/plugins/GsfTrackProducer.h"

// DEFINE_SEAL_MODULE();
DEFINE_FWK_MODULE(TrackProducer);
DEFINE_FWK_MODULE(TrackRefitter);
DEFINE_FWK_MODULE(GsfTrackProducer);
