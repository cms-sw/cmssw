#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer_Ph2.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
DEFINE_DIGI_ACCUMULATOR(EcalDigiProducer);
DEFINE_DIGI_ACCUMULATOR(EcalDigiProducer_Ph2);
DEFINE_DIGI_ACCUMULATOR(EcalTimeDigiProducer);
