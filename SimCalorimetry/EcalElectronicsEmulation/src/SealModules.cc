#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimpleSource.h"
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimRawData.h"

DEFINE_SEAL_MODULE() ;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(EcalSimpleSource);
DEFINE_ANOTHER_FWK_MODULE(EcalSimRawData);
