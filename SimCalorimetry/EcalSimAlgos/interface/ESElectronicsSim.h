#ifndef EcalSimAlgos_ESElectronicsSim_h
#define EcalSimAlgos_ESElectronicsSim_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"

#include<vector>

class ESElectronicsSim
{
 public:

  ESElectronicsSim (bool addNoise, int sigma);

  void setNoiseSigma (const int sigma) ;
  void analogToDigital(CaloSamples& clf, ESDataFrame& df) const;

  ///  anything that needs to be done once per event
  void newEvent() {}

  private :

    bool addNoise_;
    int sigma_;

    std::vector<ESSample> encode(const CaloSamples& timeframe) const;
} ;


#endif
