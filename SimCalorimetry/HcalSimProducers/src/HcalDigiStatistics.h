#ifndef HcalSimProducers_HcalDigiStatistics_h
#define HcalSimProducers_HcalDigiStatistics_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloValidationStatistics.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitAnalyzer.h"
#include <string>

class HcalDigiStatistics
{
public:
  HcalDigiStatistics(std::string name,
                     int maxBin,
                     float amplitudeThreshold,
                     float expectedPedestal,
                     float binPrevToBinMax,
                     float binNextToBinMax,
                     CaloHitAnalyzer & amplitudeAnalyzer)
:  maxBin_(maxBin),
   amplitudeThreshold_(amplitudeThreshold),
   pedestal_(name+" pedestal", expectedPedestal, 0.),
   binPrevToBinMax_(name+" binPrevToBinMax", binPrevToBinMax, 0.),
   binNextToBinMax_(name+" binNextToBinMax", binNextToBinMax, 0.),
   amplitudeAnalyzer_(amplitudeAnalyzer)
{
}

  template<class Digi>
  void analyze(const Digi & digi);

private:
  int maxBin_;
  float amplitudeThreshold_;
  CaloValidationStatistics pedestal_;
  CaloValidationStatistics binPrevToBinMax_;
  CaloValidationStatistics binNextToBinMax_;
  CaloHitAnalyzer & amplitudeAnalyzer_;
};


template<class Digi>
void HcalDigiStatistics::analyze(const Digi & digi) {
   pedestal_.addEntry(digi[0].adc());
   pedestal_.addEntry(digi[1].adc());
                                                                               
                                                                               
   double pedestal_fC = 0.5*(digi[0].nominal_fC() + digi[1].nominal_fC());
                                                                               
                                                                              
  double maxAmplitude = digi[maxBin_].nominal_fC()   - pedestal_fC;
                                                                              
  if(maxAmplitude > amplitudeThreshold_) {
                                                                              
    double binPrevToBinMax = (digi[maxBin_-1].nominal_fC() - pedestal_fC)
                           / maxAmplitude;
    binPrevToBinMax_.addEntry(binPrevToBinMax);
                                                                              
                                                                              
    double binNextToBinMax = (digi[maxBin_+1].nominal_fC() - pedestal_fC)
                           / maxAmplitude;
    binNextToBinMax_.addEntry(binNextToBinMax);
                                                                              
    double amplitude = digi[maxBin_].nominal_fC()
                   + digi[maxBin_+1].nominal_fC()
                   - 2*pedestal_fC;
                                                                                
                                                                                
    amplitudeAnalyzer_.analyze(digi.id().rawId(), amplitude);
                                                                                
  }
}

#endif

