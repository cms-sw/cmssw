#ifndef GNoiseFP420_h
#define GNoiseFP420_h

#include "SimRomanPot/SimFP420/interface/PileUpFP420.h"

// add noise
class GNoiseFP420 {
public:
  // GNoiseFP420(int,float,float);
  virtual ~GNoiseFP420() {}

  virtual PileUpFP420::signal_map_type addNoise(const PileUpFP420::signal_map_type &) = 0;
};
#endif
