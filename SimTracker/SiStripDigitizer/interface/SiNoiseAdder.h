#ifndef _TRACKER_SINOISEADDER_H
#define _TRACKER_SINOISEADDER_H

#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"

/**
 * Base class to add noise to the strips.
 */
class SiNoiseAdder{
 public:
  //SiNoiseAdder(int,float,float);
  virtual ~SiNoiseAdder() { }
  virtual SiPileUpSignals::signal_map_type addNoise(SiPileUpSignals::signal_map_type) = 0;
};
#endif
