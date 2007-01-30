#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/DTRecHits/src/DTRecHitQuality.h"
#include "Validation/DTRecHits/src/DTSegment2DQuality.h"
#include "Validation/DTRecHits/src/DTSegment2DSLPhiQuality.h"
#include "Validation/DTRecHits/src/DTSegment4DQuality.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTRecHitQuality);
DEFINE_ANOTHER_FWK_MODULE(DTSegment2DQuality);
DEFINE_ANOTHER_FWK_MODULE(DTSegment2DSLPhiQuality);
DEFINE_ANOTHER_FWK_MODULE(DTSegment4DQuality);
