#ifndef SimCalorimetry_EcalSimAlgos_EcalLiteDTUCoder_h
#define SimCalorimetry_EcalSimAlgos_EcalLiteDTUCoder_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalCATIAGainRatios.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

template <typename M>
class CorrelatedNoisifier;
class EcalLiteDTUSample;
class EcalDataFrame_Ph2;
class DetId;
class EcalLiteDTUPed;

#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class EcalLiteDTUCoder {
public:
  typedef CaloTSamples<float, ecalPh2::sampleSize> EcalSamples;

  typedef CorrelatedNoisifier<EcalCorrMatrix_Ph2> Noisifier;

  /// ctor
  EcalLiteDTUCoder(bool addNoise, bool PreMix1, Noisifier* ebCorrNoise0, Noisifier* ebCorrNoise1 = nullptr);

  /// dtor
  virtual ~EcalLiteDTUCoder();

  /// can be fetched every event from the EventSetup
  void setPedestals(const EcalLiteDTUPedestalsMap* pedestals);

  void setGainRatios(float gainRatios);

  void setFullScaleEnergy(double EBscale);

  void setIntercalibConstants(const EcalIntercalibConstantsMC* ical);

  /// from EcalSamples to EcalDataFrame_Ph2
  virtual void analogToDigital(CLHEP::HepRandomEngine*, const EcalSamples& clf, EcalDataFrame_Ph2& df) const;

private:
  /// limit on the energy scale due to the electronics range
  double fullScaleEnergy(const DetId& did) const;

  /// produce the pulse-shape
  void encode(const EcalSamples& ecalSamples, EcalDataFrame_Ph2& df, CLHEP::HepRandomEngine*) const;

  void findPedestal(const DetId& detId, int gainId, double& pedestal, double& width) const;

  void findGains(const DetId& detId, float theGains[]) const;

  void findIntercalibConstant(const DetId& detId, double& icalconst) const;

  const EcalLiteDTUPedestalsMap* m_peds;

  float m_gainRatios;  // the electronics gains

  const EcalIntercalibConstantsMC* m_intercals;  //record specific for simulation of gain variation in MC

  double m_maxEneEB;  // max attainable energy in the ecal barrel

  bool m_addNoise;  // whether add noise to the pedestals and the gains
  bool m_PreMix1;   // Follow necessary steps for PreMixing input

  const Noisifier* m_ebCorrNoise[ecalPh2::NGAINS];
};

#endif
