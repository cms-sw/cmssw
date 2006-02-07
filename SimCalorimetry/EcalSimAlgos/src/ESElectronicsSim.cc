#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"

#include "CLHEP/Random/RandGaussQ.h"

ESElectronicsSim::ESElectronicsSim (int sigma):
  m_sigma (sigma)
{
}

void ESElectronicsSim::setNoiseSigma (const int sigma)
{
  m_sigma = sigma ;
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
    // fake 1 ADC = 1 eV for the moment 
    adc = int(timeframe[i]*1000000.) + int(RandGauss::shoot(0., m_sigma));
    results.push_back(ESSample(adc));
  }

  return results;
}



