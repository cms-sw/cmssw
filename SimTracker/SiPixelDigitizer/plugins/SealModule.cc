#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SiPixelDigitizer.h"
#include <FWCore/ServiceRegistry/interface/ServiceMaker.h>
#include "SimTracker/SiPixelDigitizer/interface/RemapDetIdService.h"

using cms::SiPixelDigitizer;
DEFINE_DIGI_ACCUMULATOR(SiPixelDigitizer);

namespace simtracker { namespace services { DEFINE_FWK_SERVICE( RemapDetIdService ); } }
