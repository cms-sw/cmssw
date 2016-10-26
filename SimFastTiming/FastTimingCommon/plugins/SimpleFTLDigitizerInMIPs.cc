#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizer.h"
#include "SimFastTiming/FastTimingCommon/interface/SimpleDeviceSimInMIPs.h"
#include "SimFastTiming/FastTimingCommon/interface/SimpleElectronicsSimInMIPs.h"

typedef ftl_digitizer::FTLDigitizer<SimpleDeviceSimInMIPs,SimpleElectronicsSimInMIPs> SimpleFTLDigitizerInMIPs;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(FTLDigitizerFactory, 
		  SimpleFTLDigitizerInMIPs, 
		  "SimpleFTLDigitizerInMIPs");

