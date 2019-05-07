#ifndef EcalSimAlgos_EcalCorrelatedNoiseMatrix_h
#define EcalSimAlgos_EcalCorrelatedNoiseMatrix_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Math/interface/Error.h"

typedef math::ErrorD<CaloSamples::MAXSAMPLES>::type EcalCorrMatrix;

#endif
