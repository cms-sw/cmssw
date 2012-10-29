#ifndef _TRACKER_SINOISEADDER_H
#define _TRACKER_SINOISEADDER_H

#include "SiPileUpSignals.h"

/**
 * Base class to add noise to the strips.
 */
class SiNoiseAdder{
 public:
  virtual ~SiNoiseAdder() { }
  virtual void addNoise(std::vector<double>&,size_t&,size_t&,int,float) const = 0;
  //virtual void createRaw(std::vector<double>&,size_t&,size_t&,int,float,float) const = 0;
  
  virtual void addNoiseVR(std::vector<double> &, std::vector<float> &) const=0;
  virtual void addPedestals(std::vector<double> &, std::vector<float> &) const=0;
  virtual void addCMNoise(std::vector<double> &, float,  std::vector<bool> &) const=0;
  virtual void addBaselineShift(std::vector<double> &, std::vector<bool> &) const=0;
 
  //virtual void addNoiseVR(std::vector<double> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addPedestals(std::vector<double> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addConstNoise(std::vector<double> &, float) const=0;
  //virtual void addSingleStripNoise(std::vector<double> &, std::vector<float> &) const=0;
  //virtual void addConstPed(std::vector<double> &, float) const=0;
  //virtual void addRealPed(std::vector<double> &, std::vector<float> &) const=0;
  //virtual void addCMNoise(std::vector<double> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addBaselineShift(std::vector<double> &, std::vector<std::pair<int, float> > &) const=0;
 
};
#endif
