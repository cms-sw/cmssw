#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "CLHEP/Random/RandGaussQ.h"

using namespace std;

ESElectronicsSim::ESElectronicsSim (bool addNoise, double sigma):
  addNoise_(addNoise), sigma_ (sigma)
{
}

void ESElectronicsSim::setNoiseSigma (const double sigma)
{
  sigma_ = sigma ;
  return ;
}

void ESElectronicsSim::analogToDigital(CaloSamples& cs, ESDataFrame& df) const 
{

  std::vector<ESSample> essamples = encode(cs);

  df.setSize(cs.size());
  for(int i=0; i<df.size(); i++) {
    df.setSample(i, essamples[i]);
  }

}

std::vector<ESSample>
ESElectronicsSim::encode(const CaloSamples& timeframe) const
{

  std::vector<ESSample> results;

  int adc; 

  for (int i=0; i<timeframe.size(); i++) {
    // pedestal baseline is set to 1000
    // fake 1 ADC = 1 eV for the moment
    int pedestal = 1000;
    int noi = 0;
    if (addNoise_) noi = int(RandGauss::shoot(0., sigma_));
    adc = int(timeframe[i]*1000000.) + noi + pedestal;
    if (adc>4095) adc = 4095;
    if (adc<0) adc = 0;
    results.push_back(ESSample(adc));
  }

  return results;
}



