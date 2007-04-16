#include "TrackingTools/GsfTools/plugins/CloseComponentsMergerESProducer.h"
#include "TrackingTools/GsfTools/plugins/DistanceBetweenComponentsESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(CloseComponentsMergerESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(DistanceBetweenComponentsESProducer);

