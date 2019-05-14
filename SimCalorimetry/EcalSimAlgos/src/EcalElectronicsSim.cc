#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include "CLHEP/Random/RandGaussQ.h"

#include <cstring>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fstream>

EcalElectronicsSim::EcalElectronicsSim(const EcalSimParameterMap* parameterMap,
                                       EcalCoder* coder,
                                       bool applyConstantTerm,
                                       double rmsConstantTerm)
    : m_simMap(parameterMap), m_theCoder(coder), m_thisCT(rmsConstantTerm), m_applyConstantTerm(applyConstantTerm) {}

EcalElectronicsSim::~EcalElectronicsSim() {}

void EcalElectronicsSim::analogToDigital(CLHEP::HepRandomEngine* engine,
                                         EcalElectronicsSim::EcalSamples& clf,
                                         EcalDataFrame& df) const {
  //PG input signal is in pe.  Converted in GeV
  amplify(clf, engine);

  m_theCoder->analogToDigital(engine, clf, df);
}

void EcalElectronicsSim::amplify(EcalElectronicsSim::EcalSamples& clf, CLHEP::HepRandomEngine* engine) const {
  const double fac(m_simMap->simParameters(clf.id()).photoelectronsToAnalog());
  if (m_applyConstantTerm) {
    clf *= fac * CLHEP::RandGaussQ::shoot(engine, 1.0, m_thisCT);
  } else {
    clf *= fac;
  }
}
