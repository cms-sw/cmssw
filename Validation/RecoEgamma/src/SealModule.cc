#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoEgamma/interface/EgammaElectrons.h"
//#include "Validation/RecoEgamma/interface/EgammaPhotons.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(EgammaElectrons);
//DEFINE_ANOTHER_FWK_MODULE(EgammaPhotons);
