#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <Validation/GlobalDigis/interface/GlobalDigisProducer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalDigisProducer);

#include <Validation/GlobalDigis/interface/GlobalDigisAnalyzer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalDigisAnalyzer);

#include <Validation/GlobalDigis/interface/GlobalDigisHistogrammer.h>
DEFINE_ANOTHER_FWK_MODULE(GlobalDigisHistogrammer);
