#ifndef SimCalorimetry_EcalSimAlgos_EcalLiteDTUCoder_h
#define SimCalorimetry_EcalSimAlgos_EcalLiteDTUCoder_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalCATIAGainRatios.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/DetId/interface/DetIdFwd.h"

template <typename M>
class CorrelatedNoisifier;
class EcalLiteDTUSample;
class EcalDataFrame_Ph2;

#include <array>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class EcalLiteDTUCoder {
public:
  typedef CaloTSamples<float, ecalPh2::sampleSize> EcalSamples;

  typedef CorrelatedNoisifier<EcalCorrMatrix_Ph2> Noisifier;

  /// ctor
  EcalLiteDTUCoder(bool addNoise,
                   bool PreMix1,
                   unsigned int lowGainPreSamples,
                   unsigned int lowGainPostSamples,
                   Noisifier* ebCorrNoise0,
                   Noisifier* ebCorrNoise1 = nullptr);

  /// dtor
  virtual ~EcalLiteDTUCoder() = default;

  /// can be fetched every event from the EventSetup
  void setPedestals(const EcalLiteDTUPedestalsMap* pedestals);

  void setGainRatios(EcalCATIAGainRatios const* gainRatios);

  void setFullScaleEnergy(double EBscale);

  void setIntercalibConstants(const EcalIntercalibConstantsMC* ical);

  /// from EcalSamples to EcalDataFrame_Ph2
  virtual void analogToDigital(CLHEP::HepRandomEngine*, const EcalSamples& clf, EcalDataFrame_Ph2& df) const;

private:
  /// produce the pulse-shape
  void encode(const EcalSamples& ecalSamples, EcalDataFrame_Ph2& df, CLHEP::HepRandomEngine*) const;

  void findPedestal(const DetId& detId, int gainId, double& pedestal, double& width) const;

  void findGains(const DetId& detId, std::array<double, ecalPh2::NGAINS>& gains) const;

  void findIntercalibConstant(const DetId& detId, double& icalconst) const;

  const EcalLiteDTUPedestalsMap* m_peds;

  EcalCATIAGainRatios const* m_gainRatios;  // the electronics gain ratios

  const EcalIntercalibConstantsMC* m_intercals;  //record specific for simulation of gain variation in MC

  double m_maxEneEB;  // max attainable energy in the ecal barrel

  bool const m_addNoise;                    // whether add noise to the pedestals and the gains
  bool const m_PreMix1;                     // Follow necessary steps for PreMixing input
  unsigned int const m_lowGainPreSamples;   // number of samples in low gain before the first gain switched sample
  unsigned int const m_lowGainPostSamples;  // number of samples in low gain after the last gain switched sample

  static constexpr double m_quantRMS2 = 1. / 12.;  // square of quantization error RMS 1/sqrt(12)

  std::array<Noisifier*, ecalPh2::NGAINS> const m_ebCorrNoise;
};

#endif
