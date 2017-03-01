#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"

DEFINE_DIGI_ACCUMULATOR(EcalDigiProducer);
DEFINE_DIGI_ACCUMULATOR(EcalTimeDigiProducer);

