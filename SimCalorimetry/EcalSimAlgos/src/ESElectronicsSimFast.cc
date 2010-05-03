#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGeneral.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

ESElectronicsSimFast::ESElectronicsSimFast (bool addNoise, double sigma, int gain, int baseline, double MIPADC, double MIPkeV):
  addNoise_(addNoise), sigma_ (sigma), gain_ (gain), baseline_(baseline), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{
  // Preshower "Fast" Electronics Simulation
  // The default pedestal baseline is 1000
  // gain = 0 : old gain used in ORCA (1 ADC count = 1 keV in CMSSW)
  //            In ORCA, preshower noise was 15 keV
  // gain = 1 : low gain for data taking  (S =  9 ADC counts, N = 3 ADC counts)
  // gain = 2 : high gain for calibration (S = 50 ADC counts, N = 7 ADC counts)
  // For 300(310/320) um Si, the MIP is 78.47(81.08/83.7) keV
}

void ESElectronicsSimFast::setNoiseSigma (const double sigma)
{
  sigma_ = sigma ;
  return ;
}

void ESElectronicsSimFast::setGain (const int gain)
{
  gain_ = gain ;
  return ;
}

void ESElectronicsSimFast::setBaseline (const int baseline)
{
  baseline_ = baseline ;
  return ;
}

void ESElectronicsSimFast::setMIPADC (const double MIPADC) 
{
  MIPADC_ = MIPADC ;
  return ;
}

void ESElectronicsSimFast::setMIPkeV (const double MIPkeV) 
{
  MIPkeV_ = MIPkeV ;
  return ;
}

void ESElectronicsSimFast::analogToDigital(const CaloSamples& cs, ESDataFrame& df, bool wasEmpty, CLHEP::RandGeneral *histoDistribution, double hInf, double hSup, double hBin) const 
{
  std::vector<ESSample> essamples;
  if (!wasEmpty) essamples = standEncode(cs);
  if ( wasEmpty) essamples = fastEncode(cs, histoDistribution, hInf, hSup, hBin);
  
  df.setSize(cs.size());
  for(int i=0; i<df.size(); i++) {
    df.setSample(i, essamples[i]);
  }
}

void ESElectronicsSimFast::digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const 
{
  for(int i = 0; i < df.size(); i++) {
    cs[i] = decode(df[i], df.id());
  } 
}

std::vector<ESSample>
ESElectronicsSimFast::standEncode(const CaloSamples& timeframe) const
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "ESElectroncSimFast requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  std::vector<ESSample> results;
  results.reserve(timeframe.size());
  
  int adc = 0; 
  double ADCkeV = MIPADC_/MIPkeV_;
  for (int i=0; i<timeframe.size(); i++) {
    
    double noi = 0;
    double signal = 0;    
    
    if (addNoise_) {
      CLHEP::RandGaussQ gaussQDistribution(rng->getEngine(), 0, sigma_);
      noi = gaussQDistribution.fire();
    }
    
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


std::vector<ESSample>
ESElectronicsSimFast::fastEncode(const CaloSamples& timeframe, CLHEP::RandGeneral *histoDistribution, double hInf, double hSup, double hBin) const
{
  std::vector<ESSample> results;
  results.reserve(timeframe.size());

  int bin[3]; 
  double hBin2 = hBin*hBin;
  double hBin3 = hBin*hBin*hBin;
  double width = (hSup - hInf)/hBin;  

  double thisRnd  = histoDistribution->fire();  
  int thisRndCell = (int)((hBin3)*(thisRnd)/width);  
  bin[2] = (int)(thisRndCell/hBin2);                              // sample2 - bin [0,N-1]
  bin[1] = (int)((thisRndCell - hBin2*bin[2])/hBin);              // sample1
  bin[0] = (int)(thisRndCell - hBin*(bin[1] + hBin*bin[2]));      // sample0

  int adc[3];
  double noi[3];
  for(int ii=0; ii<3; ii++){

    noi[ii] = hInf + bin[ii]*width; 
    if (noi[ii]>0)      noi[ii] += 0.5;
    else if (noi[ii]<0) noi[ii] -= 0.5;
    
    adc[ii] = int(noi[ii]);      
    if (adc[ii]>MAXADC) adc[ii] = MAXADC;
    if (adc[ii]<MINADC) adc[ii] = MINADC;
    
    results.push_back(ESSample(adc[ii]));
  }
  
  return results;
}

double ESElectronicsSimFast::decode(const ESSample & sample, const DetId & id) const
{
  return 0. ;
}



