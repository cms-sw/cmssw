#ifndef EcalSimAlgos_EcalCorrelatedNoiseMatrix_h
#define EcalSimAlgos_EcalCorrelatedNoiseMatrix_h

#include "DataFormats/Math/interface/Error.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

typedef math::ErrorD<ecalPh1::sampleSize>::type EcalCorrMatrix;
typedef math::ErrorD<ecalPh2::sampleSize>::type EcalCorrMatrix_Ph2;
#endif
