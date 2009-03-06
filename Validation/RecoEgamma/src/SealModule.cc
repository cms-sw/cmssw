#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "Validation/RecoEgamma/interface/EgammaObjects.h"
#include "Validation/RecoEgamma/interface/PhotonValidator.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(EgammaObjects);
DEFINE_ANOTHER_FWK_MODULE(PhotonValidator);

