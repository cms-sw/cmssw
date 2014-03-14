#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"

DEFINE_DIGI_ACCUMULATOR(EcalDigiProducer);

#include "SimCalorimetry/EcalSimProducers/interface/EcalPhaseIIDigiProducer.h"
DEFINE_DIGI_ACCUMULATOR(EcalPhaseIIDigiProducer);

#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
DEFINE_DIGI_ACCUMULATOR(EcalTimeDigiProducer);


