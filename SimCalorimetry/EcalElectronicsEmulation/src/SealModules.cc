#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalFEtoDigi.h"
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimRawData.h"
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimpleProducer.h"

DEFINE_FWK_MODULE(EcalSimpleProducer);
DEFINE_FWK_MODULE(EcalSimRawData);
DEFINE_FWK_MODULE(EcalFEtoDigi);
