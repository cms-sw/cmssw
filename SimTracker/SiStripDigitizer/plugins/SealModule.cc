#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizer.h"
#include "SimTracker/SiStripDigitizer/interface/DigiSimLinkProducer.h"

DEFINE_DIGI_ACCUMULATOR(SiStripDigitizer);
DEFINE_FWK_MODULE(DigiSimLinkProducer);

