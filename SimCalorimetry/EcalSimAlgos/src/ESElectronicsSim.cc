#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

ESElectronicsSim::ESElectronicsSim (bool addNoise):
  addNoise_(addNoise), peds_(0), mips_(0)
{
  // Preshower Electronics Simulation
  // gain = 1 : low gain for data taking 
  // gain = 2 : high gain for calibration and low energy runs
  // For 300(310/320) um Si, the MIP is 78.47(81.08/83.7) keV
}

ESElectronicsSim::~ESElectronicsSim ()
{}

void ESElectronicsSim::analogToDigital(const CaloSamples& cs, ESDataFrame& df) const 
{

  std::vector<ESSample> essamples = encode(cs);

  df.setSize(cs.size());
  for(int i=0; i<df.size(); i++) {
    df.setSample(i, essamples[i]);
  }

}

void ESElectronicsSim::digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const 
{

  for(int i = 0; i < df.size(); i++) {
    cs[i] = decode(df[i], df.id());
  }

}

std::vector<ESSample>
ESElectronicsSim::encode(const CaloSamples& timeframe) const
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "ESElectroncSim requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }


  std::vector<ESSample> results;
  results.reserve(timeframe.size());

  ESPedestals::const_iterator it_ped = peds_->find(timeframe.id());
  ESIntercalibConstantMap::const_iterator it_mip = mips_->getMap().find(timeframe.id());
  int baseline_  = (int) it_ped->getMean();
  double sigma_  = (double) it_ped->getRms();
  double MIPADC_ = (double) (*it_mip);

  int adc = 0; 
  double ADCGeV = MIPADC_/MIPToGeV_;

  for (int i=0; i<timeframe.size(); i++) {

    double noi = 0;
    double signal = 0;    

    if (addNoise_) {
      CLHEP::RandGaussQ gaussQDistribution(rng->getEngine(), 0., sigma_);
      noi = gaussQDistribution.fire();
    }

    signal = timeframe[i]*ADCGeV + noi + baseline_;
    
    if (signal>0) 
      signal += 0.5;
    else if (signal<0)
      signal -= 0.5;
    
    adc = int(signal);

    if (adc>MAXADC) adc = MAXADC;
    if (adc<MINADC) adc = MINADC;

    results.emplace_back(adc);
  }

  return results;
}

double ESElectronicsSim::decode(const ESSample & sample, const DetId & id) const
{
  return 0. ;
}



