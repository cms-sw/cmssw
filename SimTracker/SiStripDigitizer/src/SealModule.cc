
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizer.h"


using cms::SiStripDigitizer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripDigitizer);


