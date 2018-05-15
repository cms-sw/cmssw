#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizer.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"


typedef mtd_digitizer::MTDDigitizer<BTLDeviceSim,BTLElectronicsSim> BTLDigitizer;
typedef mtd_digitizer::MTDDigitizer<ETLDeviceSim,ETLElectronicsSim> ETLDigitizer;


DEFINE_EDM_PLUGIN(MTDDigitizerFactory, BTLDigitizer, "BTLDigitizer");
DEFINE_EDM_PLUGIN(MTDDigitizerFactory, ETLDigitizer, "ETLDigitizer");
