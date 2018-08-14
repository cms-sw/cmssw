#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizer.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTraits.h"

typedef mtd_digitizer::MTDDigitizer<BTLDigitizerTraits> BTLDigitizer;
typedef mtd_digitizer::MTDDigitizer<ETLDigitizerTraits> ETLDigitizer;


DEFINE_EDM_PLUGIN(MTDDigitizerFactory, BTLDigitizer, "BTLDigitizer");
DEFINE_EDM_PLUGIN(MTDDigitizerFactory, ETLDigitizer, "ETLDigitizer");
