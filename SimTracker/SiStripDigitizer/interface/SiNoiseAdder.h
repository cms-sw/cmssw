#ifndef _TRACKER_SINOISEADDER_H
#define _TRACKER_SINOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

/**
 * Base class to add noise to the strips.
 */
class SiNoiseAdder{
 public:
  virtual ~SiNoiseAdder() { }
  virtual void addNoise(std::vector<double>&,size_t&,size_t&,int,float) = 0;
  //virtual void createRaw(std::vector<double>&,size_t&,size_t&,int,float,float) = 0;
  
  virtual void addNoiseVR(std::vector<double> &, std::vector<float> &)=0;
  virtual void addPedestals(std::vector<double> &, std::vector<float> &)=0;
  virtual void addCMNoise(std::vector<double> &, float,  std::vector<bool> &)=0;
  virtual void addBaselineShift(std::vector<double> &, std::vector<bool> &)=0;
 
  //virtual void addNoiseVR(std::vector<double> &, std::vector<std::pair<int, float> > &)=0;
  //virtual void addPedestals(std::vector<double> &, std::vector<std::pair<int, float> > &)=0;
  //virtual void addConstNoise(std::vector<double> &, float)=0;
  //virtual void addSingleStripNoise(std::vector<double> &, std::vector<float> &)=0;
  //virtual void addConstPed(std::vector<double> &, float)=0;
  //virtual void addRealPed(std::vector<double> &, std::vector<float> &)=0;
  //virtual void addCMNoise(std::vector<double> &, std::vector<std::pair<int, float> > &)=0;
  //virtual void addBaselineShift(std::vector<double> &, std::vector<std::pair<int, float> > &)=0;
 
};
#endif
