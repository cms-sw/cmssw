#ifndef _TRACKER_SINOISEADDER_H
#define _TRACKER_SINOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

/**
 * Base class to add noise to the strips.
 */
class SiNoiseAdder{
 public:
  virtual ~SiNoiseAdder() { }
  virtual void addNoise(SiPileUpSignals::signal_map_type &,int,float) = 0;
  virtual void createRaw(SiPileUpSignals::signal_map_type &,int,float) = 0;
};
#endif
