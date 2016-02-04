#include "TrackingTools/GsfTools/plugins/CloseComponentsMergerESProducer.h"
#include "TrackingTools/GsfTools/plugins/DistanceBetweenComponentsESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

// DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsMergerESProducer);
// DEFINE_FWK_EVENTSETUP_MODULE(DistanceBetweenComponentsESProducer);

typedef CloseComponentsMergerESProducer<5> CloseComponentsMergerESProducer5D;
typedef DistanceBetweenComponentsESProducer<5> DistanceBetweenComponentsESProducer5D;

DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsMergerESProducer5D);
DEFINE_FWK_EVENTSETUP_MODULE(DistanceBetweenComponentsESProducer5D);

//   KullbackLeiblerDistance<5> kbd;
