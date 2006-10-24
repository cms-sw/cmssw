#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/TrackerRecHits/interface/SiStripRecHitsValid.h"
#include "Validation/TrackerRecHits/interface/SiPixelRecHitsValid.h"

DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE(SiStripRecHitsValid);
DEFINE_ANOTHER_FWK_MODULE(SiPixelRecHitsValid);
