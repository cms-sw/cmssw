#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MultiTrackValidator);
DEFINE_ANOTHER_FWK_MODULE(SiStripTrackingRecHitsValid);

