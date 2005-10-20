
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimTracker/SiPixelDigitizer/interface/SiPixelDigitizer.h"


using cms::SiPixelDigitizer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiPixelDigitizer)


