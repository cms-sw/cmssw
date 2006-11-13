#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "CLHEP/Random/RandGaussQ.h"

using namespace std;

ESElectronicsSim::ESElectronicsSim (bool addNoise, double sigma, int gain, int baseline, double MIPADC, double MIPkeV):
  addNoise_(addNoise), sigma_ (sigma), gain_ (gain), baseline_(baseline), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{
  // Preshower Electronics Simulation
  // The default pedestal baseline is 1000
  // gain = 0 : old gain used in ORCA (1 ADC count = 1 keV in CMSSW)
  //            In ORCA, preshower noise was 15 keV
  // gain = 1 : low gain for data taking  (S =  9 ADC counts, N = 3 ADC counts)
  // gain = 2 : high gain for calibration (S = 50 ADC counts, N = 7 ADC counts)
  // For 300(310/320) um Si, the MIP is 78.47(81.08/83.7) keV
}

void ESElectronicsSim::setNoiseSigma (const double sigma)
{
  sigma_ = sigma ;
  return ;
}

void ESElectronicsSim::setGain (const int gain)
{
  gain_ = gain ;
  return ;
}

void ESElectronicsSim::setBaseline (const int baseline)
{
  baseline_ = baseline ;
  return ;
}

void ESElectronicsSim::setMIPADC (const double MIPADC) 
{
  MIPADC_ = MIPADC ;
  return ;
}

void ESElectronicsSim::setMIPkeV (const double MIPkeV) 
{
  MIPkeV_ = MIPkeV ;
  return ;
}

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

  std::vector<ESSample> results;
  results.reserve(timeframe.size());

  int adc = 0; 
  double ADCkeV = MIPADC_/MIPkeV_;

  for (int i=0; i<timeframe.size(); i++) {

    double noi = 0;
    double signal = 0;    

    if (addNoise_) noi = RandGaussQ::shoot(0., sigma_);

    if (gain_ == 0) { 
      signal = timeframe[i]*1000000. + noi + baseline_;     

      if (signal>0) 
	signal += 0.5;
      else if (signal<0)
	signal -= 0.5;

      adc = int(signal);
    }
    else if (gain_ == 1 || gain_ == 2) {
      signal = timeframe[i]*1000000.*ADCkeV + noi + baseline_;

      if (signal>0) 
	signal += 0.5;
      else if (signal<0)
	signal -= 0.5;

      adc = int(signal);
    }

    if (adc>MAXADC) adc = MAXADC;
    if (adc<MINADC) adc = MINADC;

    results.push_back(ESSample(adc));
  }

  return results;
}

double ESElectronicsSim::decode(const ESSample & sample, const DetId & id) const
{
  return 0. ;
}



