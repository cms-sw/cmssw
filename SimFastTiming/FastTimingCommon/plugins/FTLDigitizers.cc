#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizer.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"


typedef ftl_digitizer::FTLDigitizer<BTLDeviceSim,BTLElectronicsSim> BTLDigitizer;
typedef ftl_digitizer::FTLDigitizer<ETLDeviceSim,ETLElectronicsSim> ETLDigitizer;


DEFINE_EDM_PLUGIN(FTLDigitizerFactory, BTLDigitizer, "BTLDigitizer");
DEFINE_EDM_PLUGIN(FTLDigitizerFactory, ETLDigitizer, "ETLDigitizer");
