#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/DTRecHits/plugins/DTRecHitQuality.h"
#include "Validation/DTRecHits/plugins/DTSegment2DQuality.h"
#include "Validation/DTRecHits/plugins/DTSegment2DSLPhiQuality.h"
#include "Validation/DTRecHits/plugins/DTSegment4DQuality.h"

#include "Validation/DTRecHits/plugins/DTRecHitClients.h"
#include "Validation/DTRecHits/plugins/DT2DSegmentClients.h"
#include "Validation/DTRecHits/plugins/DT4DSegmentClients.h"


DEFINE_FWK_MODULE(DTRecHitQuality);
DEFINE_FWK_MODULE(DTSegment2DQuality);
DEFINE_FWK_MODULE(DTSegment2DSLPhiQuality);
DEFINE_FWK_MODULE(DTSegment4DQuality);

DEFINE_FWK_MODULE(DTRecHitClients);
DEFINE_FWK_MODULE(DT2DSegmentClients);
DEFINE_FWK_MODULE(DT4DSegmentClients);
