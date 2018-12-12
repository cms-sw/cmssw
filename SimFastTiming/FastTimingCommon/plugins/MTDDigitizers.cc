#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizer.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTraits.h"

typedef mtd_digitizer::MTDDigitizer<BTLTileDigitizerTraits> BTLTileDigitizer;
typedef mtd_digitizer::MTDDigitizer<BTLBarDigitizerTraits> BTLBarDigitizer;
typedef mtd_digitizer::MTDDigitizer<ETLDigitizerTraits> ETLDigitizer;


DEFINE_EDM_PLUGIN(MTDDigitizerFactory, BTLTileDigitizer, "BTLTileDigitizer");
DEFINE_EDM_PLUGIN(MTDDigitizerFactory, BTLBarDigitizer, "BTLBarDigitizer");
DEFINE_EDM_PLUGIN(MTDDigitizerFactory, ETLDigitizer, "ETLDigitizer");
