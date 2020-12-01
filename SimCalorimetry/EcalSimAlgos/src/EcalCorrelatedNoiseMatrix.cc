#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.icc"

template class CorrelatedNoisifier<EcalCorrMatrix>;
template class CorrelatedNoisifier<EcalCorrMatrix_Ph2>;

template void CorrelatedNoisifier<EcalCorrMatrix>::noisify(CaloSamples&,
                                                           CLHEP::HepRandomEngine*,
                                                           const std::vector<double>* rangau) const;

template void CorrelatedNoisifier<EcalCorrMatrix_Ph2>::noisify(CaloSamples&,
                                                               CLHEP::HepRandomEngine*,
                                                               const std::vector<double>* rangau) const;
