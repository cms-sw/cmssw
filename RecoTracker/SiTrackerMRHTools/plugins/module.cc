#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/SiTrackerMRHTools/plugins/SiTrackerMultiRecHitUpdatorESProducer.h"
#include "RecoTracker/SiTrackerMRHTools/plugins/MultiRecHitCollectorESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiTrackerMultiRecHitUpdatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(MultiRecHitCollectorESProducer);
