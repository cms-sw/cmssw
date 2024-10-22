#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.icc"

template class EcalTDigitizer<EBDigitizerTraits>;
template class EcalTDigitizer<EEDigitizerTraits>;
template class EcalTDigitizer<ESDigitizerTraits>;

template class EcalTDigitizer<EBDigitizerTraits_Ph2>;
