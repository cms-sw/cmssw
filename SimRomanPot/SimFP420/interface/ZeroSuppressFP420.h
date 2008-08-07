#ifndef ZeroSuppressFP420_h
#define ZeroSuppressFP420_h

#include "SimRomanPot/SimFP420/interface/ZSuppressFP420.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ZeroSuppressFP420 : public ZSuppressFP420{
 public:
   
  /* Read  the noise in the channels.*/
  ZeroSuppressFP420(const edm::ParameterSet& conf, float noise);
  virtual ~ZeroSuppressFP420() {}

  /* calculates the lower and high signal thresholds using the noise */
  void initParams(const edm::ParameterSet& conf_);
 
  ZSuppressFP420::DigitalMapType zeroSuppress(const DigitalMapType&,int);
  
  ZSuppressFP420::DigitalMapType trkFEDclusterizer(const DigitalMapType&,int); 
  
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
