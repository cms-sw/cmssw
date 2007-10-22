#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/GlobalRecHits/interface/GlobalRecHitsProducer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalRecHitsProducer);

#include <Validation/GlobalRecHits/interface/GlobalRecHitsAnalyzer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalRecHitsAnalyzer);

#include <Validation/GlobalRecHits/interface/GlobalRecHitsHistogrammer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalRecHitsHistogrammer);
