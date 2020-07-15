#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TrackProducer/plugins/DAFTrackProducer.h"
#include "RecoTracker/TrackProducer/plugins/TrackProducer.h"
#include "RecoTracker/TrackProducer/plugins/TrackRefitter.h"
#include "RecoTracker/TrackProducer/plugins/GsfTrackRefitter.h"
#include "RecoTracker/TrackProducer/plugins/ExtraFromSeeds.h"
#include "RecoTracker/TrackProducer/plugins/TrackingRecHitThinningProducer.h"
#include "RecoTracker/TrackProducer/plugins/SiPixelClusterThinningProducer.h"
#include "RecoTracker/TrackProducer/plugins/SiStripClusterThinningProducer.h"

//
DEFINE_FWK_MODULE(DAFTrackProducer);
DEFINE_FWK_MODULE(TrackProducer);
DEFINE_FWK_MODULE(TrackRefitter);
DEFINE_FWK_MODULE(GsfTrackRefitter);
DEFINE_FWK_MODULE(ExtraFromSeeds);
DEFINE_FWK_MODULE(TrackingRecHitThinningProducer);
DEFINE_FWK_MODULE(SiPixelClusterThinningProducer);
DEFINE_FWK_MODULE(SiStripClusterThinningProducer);
