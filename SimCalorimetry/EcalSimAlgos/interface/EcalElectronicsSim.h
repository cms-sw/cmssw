#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

class EcalSimParameterMap;

namespace CLHEP {
  class HepRandomEngine;
}

/* \class EcalElectronicsSim
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 * 
 */

template <typename CoderType, typename SamplesType, typename DataFrameType>
class EcalElectronicsSim {
public:
  EcalElectronicsSim(const EcalSimParameterMap* parameterMap,
                     CoderType* coder,
                     bool applyConstantTerm,
                     double rmsConstantTerm)
      : m_simMap(parameterMap), m_theCoder(coder), m_thisCT(rmsConstantTerm), m_applyConstantTerm(applyConstantTerm) {}

  /// from EcalSamples to EcalDataFrame

  void analogToDigital(CLHEP::HepRandomEngine* engine, SamplesType& clf, DataFrameType& df) const {
    // input signal is in pe.  Converted in GeV
    amplify(clf, engine);
    m_theCoder->analogToDigital(engine, clf, df);
  }

  void newEvent() {}

  void setNoiseSignalGenerator(const CaloVNoiseSignalGenerator* noiseSignalGenerator) {
    theNoiseSignalGenerator = noiseSignalGenerator;
  }

private:
  /// input signal is in pe.  Converted in GeV
  void amplify(SamplesType& clf, CLHEP::HepRandomEngine* engine) const {
    const double fac(m_simMap->simParameters(clf.id()).photoelectronsToAnalog());
    if (m_applyConstantTerm) {
      clf *= fac * CLHEP::RandGaussQ::shoot(engine, 1.0, m_thisCT);
    } else {
      clf *= fac;
    }
  }

  /// map of parameters

  const EcalSimParameterMap* m_simMap;

  const CaloVNoiseSignalGenerator* theNoiseSignalGenerator;

  CoderType* m_theCoder;

  const double m_thisCT;
  const bool m_applyConstantTerm;
};

#endif
