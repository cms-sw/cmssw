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
  virtual void createRaw(std::vector<double>&,size_t&,size_t&,int,float,float) = 0;
};
#endif
