#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/src/GenericTransientTrackingRecHitBuilderESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(TransientTrackingRecHitBuilder);
DEFINE_FWK_EVENTSETUP_MODULE(GenericTransientTrackingRecHitBuilderESProducer)


