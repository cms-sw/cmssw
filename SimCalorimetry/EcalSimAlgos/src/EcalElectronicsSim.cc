#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string.h>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fstream>

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
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "EcalElectroncSim requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }

  double thisCT = rmsConstantTerm_;
  CLHEP::RandGaussQ gaussQDistribution(rng->getEngine(), 0.0, thisCT);
  return gaussQDistribution.fire();
}

void EcalElectronicsSim::analogToDigital(CaloSamples& clf, EcalDataFrame& df) const 
{
  //PG input signal is in pe.  Converted in GeV
  amplify(clf);
  theCoder->analogToDigital(clf, df);
}




