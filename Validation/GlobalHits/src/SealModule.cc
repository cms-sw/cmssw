#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/GlobalHits/interface/GlobalHitsProducer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalHitsProducer);

#include <Validation/GlobalHits/interface/GlobalHitsAnalyzer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalHitsAnalyzer);
