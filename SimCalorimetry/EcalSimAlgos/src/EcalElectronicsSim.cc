#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"


EcalElectronicsSim::EcalElectronicsSim(const EcalSimParameterMap * parameterMap, 
                                       EcalCoder * coder)
: theParameterMap(parameterMap),
  theCoder(coder)
{
}


void EcalElectronicsSim::amplify(CaloSamples & clf) const 
{
  clf *= theParameterMap->simParameters(clf.id()).photoelectronsToAnalog();
}


void EcalElectronicsSim::analogToDigital(CaloSamples& clf, EBDataFrame& df) const 
{
  // input signal is in pe.  We want it in fC
  amplify(clf);
  theCoder->analogToDigital(clf, df);
}


void EcalElectronicsSim::analogToDigital(CaloSamples& clf, EEDataFrame& df) const
{
  // input signal is in pe.  We want it in fC
  amplify(clf);
  theCoder->analogToDigital(clf, df);
}


