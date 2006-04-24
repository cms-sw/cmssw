#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include "CLHEP/Random/RandGaussQ.h"

EcalElectronicsSim::EcalElectronicsSim(const EcalSimParameterMap * parameterMap, 
                                       EcalCoder * coder, 
                                       bool applyConstantTerm, 
                                       double rmsConstantTerm)
: theParameterMap(parameterMap),
  theCoder(coder),
  applyConstantTerm_(applyConstantTerm), 
  rmsConstantTerm_(rmsConstantTerm)
{
}


void EcalElectronicsSim::amplify(CaloSamples & clf) const 
{
  clf *= theParameterMap->simParameters(clf.id()).photoelectronsToAnalog();
  if (applyConstantTerm_) {
    clf *= (1.+constantTerm());
  }
}

double EcalElectronicsSim::constantTerm() const
{
  double thisCT = rmsConstantTerm_;
  return RandGaussQ::shoot(0.0,thisCT);
}

void EcalElectronicsSim::analogToDigital(CaloSamples& clf, EBDataFrame& df) const 
{
  //PG input signal is in pe.  Converted in GeV
  amplify(clf);
  theCoder->analogToDigital(clf, df);
}


void EcalElectronicsSim::analogToDigital(CaloSamples& clf, EEDataFrame& df) const
{
  // input signal is in pe.  We want it in fC
  amplify(clf);
  theCoder->analogToDigital(clf, df);
}


