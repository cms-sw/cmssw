#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTracker/SingleTrackPattern/test/ReadCosmicTracks.h"
#include "RecoTracker/SingleTrackPattern/test/AnalyzeMTCCTracks.h"
#include "RecoTracker/SingleTrackPattern/test/CombTrack.h"
#include "RecoTracker/SingleTrackPattern/test/AnalyzeHitEff.h"
#include "RecoTracker/SingleTrackPattern/test/LayerFilter.h"


DEFINE_FWK_MODULE(ReadCosmicTracks);
DEFINE_FWK_MODULE(AnalyzeMTCCTracks);
DEFINE_FWK_MODULE(CombTrack);
DEFINE_FWK_MODULE(AnalyzeHitEff);
DEFINE_FWK_MODULE(LayerFilter);
