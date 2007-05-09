#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SeedGeneratorFromRegionHitsEDProducer.h"
#include "SeedGeneratorFromProtoTracksEDProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromRegionHitsEDProducer);
DEFINE_ANOTHER_FWK_MODULE(SeedGeneratorFromProtoTracksEDProducer);
