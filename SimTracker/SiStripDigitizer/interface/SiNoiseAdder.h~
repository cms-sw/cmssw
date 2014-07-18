#ifndef _TRACKER_SINOISEADDER_H
#define _TRACKER_SINOISEADDER_H

#include "SiPileUpSignals.h"

namespace CLHEP {
  class HepRandomEngine;
}

/**
 * Base class to add noise to the strips.
 */
class SiNoiseAdder{
 public:
  virtual ~SiNoiseAdder() { }
  virtual void addNoise(std::vector<float>&,size_t&,size_t&,int,float, CLHEP::HepRandomEngine*) const = 0;
  //virtual void createRaw(std::vector<float>&,size_t&,size_t&,int,float,float) const = 0;
  
  virtual void addNoiseVR(std::vector<float> &, std::vector<float> &, CLHEP::HepRandomEngine*) const=0;
  virtual void addPedestals(std::vector<float> &, std::vector<float> &) const=0;
  virtual void addCMNoise(std::vector<float> &, float,  std::vector<bool> &, CLHEP::HepRandomEngine*) const=0;
  virtual void addBaselineShift(std::vector<float> &, std::vector<bool> &) const=0;
 
  //virtual void addNoiseVR(std::vector<float> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addPedestals(std::vector<float> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addConstNoise(std::vector<float> &, float) const=0;
  //virtual void addSingleStripNoise(std::vector<float> &, std::vector<float> &) const=0;
  //virtual void addConstPed(std::vector<float> &, float) const=0;
  //virtual void addRealPed(std::vector<float> &, std::vector<float> &) const=0;
  //virtual void addCMNoise(std::vector<float> &, std::vector<std::pair<int, float> > &) const=0;
  //virtual void addBaselineShift(std::vector<float> &, std::vector<std::pair<int, float> > &) const=0;
 
};
#endif
