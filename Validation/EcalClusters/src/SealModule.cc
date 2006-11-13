#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"
#include "Validation/EcalClusters/interface/EgammaSuperClusters.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(EgammaBasicClusters);
DEFINE_ANOTHER_FWK_MODULE(EgammaSuperClusters);
