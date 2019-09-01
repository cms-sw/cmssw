#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.icc"

template class CorrelatedNoisifier<EcalCorrMatrix>;

template void CorrelatedNoisifier<EcalCorrMatrix>::noisify(CaloSamples&,
                                                           CLHEP::HepRandomEngine*,
                                                           const std::vector<double>* rangau) const;
