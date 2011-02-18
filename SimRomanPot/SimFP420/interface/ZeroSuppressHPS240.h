#ifndef ZeroSuppressHPS240_h
#define ZeroSuppressHPS240_h

#include "SimRomanPot/SimFP420/interface/ZSuppressHPS240.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ZeroSuppressHPS240 : public ZSuppressHPS240{
 public:
   
  /* Read  the noise in the channels.*/
  ZeroSuppressHPS240(const edm::ParameterSet& conf, float noise);
  virtual ~ZeroSuppressHPS240() {}

  /* calculates the lower and high signal thresholds using the noise */
  void initParams(const edm::ParameterSet& conf_);
 
  ZSuppressHPS240::DigitalMapType zeroSuppress(const DigitalMapType&,int);
  
  ZSuppressHPS240::DigitalMapType trkFEDclusterizer(const DigitalMapType&,int); 
  
 private:
  float noiseInAdc;
  short theFEDalgorithm;
  float theFEDlowThresh;
  float theFEDhighThresh;
  edm::ParameterSet conf_;
  short theNumFEDalgos;

  int algoConf, verbosity;
  double lowthreshConf;
  double highthreshConf;
};
 
#endif
