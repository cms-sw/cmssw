#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "Validation/RecoEgamma/plugins/EgammaObjects.h"
#include "Validation/RecoEgamma/plugins/PhotonValidator.h"
#include "Validation/RecoEgamma/plugins/TkConvValidator.h"
#include "Validation/RecoEgamma/plugins/ConversionPostprocessing.h"
#include "Validation/RecoEgamma/plugins/PhotonPostprocessing.h"
#include "Validation/RecoEgamma/plugins/ElectronMcSignalValidator.h"
#include "Validation/RecoEgamma/plugins/ElectronMcSignalPostValidator.h"
#include "Validation/RecoEgamma/plugins/ElectronMcFakeValidator.h"
#include "Validation/RecoEgamma/plugins/ElectronMcFakePostValidator.h"
#include "Validation/RecoEgamma/plugins/ElectronWebGetter.h"

DEFINE_FWK_MODULE(EgammaObjects);
DEFINE_FWK_MODULE(PhotonValidator);
DEFINE_FWK_MODULE(TkConvValidator);
DEFINE_FWK_MODULE(PhotonPostprocessing);
DEFINE_FWK_MODULE(ConversionPostprocessing);
DEFINE_FWK_MODULE(ElectronMcSignalValidator);
DEFINE_FWK_MODULE(ElectronMcSignalPostValidator);
DEFINE_FWK_MODULE(ElectronMcFakeValidator);
DEFINE_FWK_MODULE(ElectronMcFakePostValidator);
DEFINE_FWK_MODULE(ElectronWebGetter);

