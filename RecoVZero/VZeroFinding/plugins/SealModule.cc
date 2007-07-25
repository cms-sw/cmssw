#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

// Producer
#include "VZeroProducer.h"
DEFINE_ANOTHER_FWK_MODULE(VZeroProducer);

