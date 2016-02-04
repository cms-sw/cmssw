#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "Validation/RecoEgamma/interface/EgammaObjects.h"
#include "Validation/RecoEgamma/interface/PhotonValidator.h"
#include "Validation/RecoEgamma/interface/TkConvValidator.h"
#include "Validation/RecoEgamma/interface/ConversionPostprocessing.h"
#include "Validation/RecoEgamma/interface/PhotonPostprocessing.h"
#include "Validation/RecoEgamma/interface/ElectronMcSignalValidator.h"
#include "Validation/RecoEgamma/interface/ElectronMcFakeValidator.h"



DEFINE_FWK_MODULE(EgammaObjects);
DEFINE_FWK_MODULE(PhotonValidator);
DEFINE_FWK_MODULE(TkConvValidator);
DEFINE_FWK_MODULE(PhotonPostprocessing);
DEFINE_FWK_MODULE(ConversionPostprocessing);
DEFINE_FWK_MODULE(ElectronMcSignalValidator);
DEFINE_FWK_MODULE(ElectronMcFakeValidator);

