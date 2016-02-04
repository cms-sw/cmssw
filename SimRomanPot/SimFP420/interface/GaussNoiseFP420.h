#ifndef GaussNoiseFP420_h
#define GaussNoiseFP420_h

#include "SimRomanPot/SimFP420/interface/GNoiseFP420.h"

// add noise for channels only with amplitudes greater  threshold
class GaussNoiseFP420 : public GNoiseFP420{
 public:
  // GaussNoiseFP420(int,float,float,bool);
  GaussNoiseFP420(int ns, float nrms, float th, bool aNpixel, int verbosity);
  PileUpFP420::signal_map_type addNoise(PileUpFP420::signal_map_type);
  void setNumPixels(int in){numPixels = in;}
  void setThreshold(float in){threshold = in;}

 private:
  int numPixels;
  float noiseRMS;
  float threshold;
  bool addNoisyPixels;
  int verbosi;
};
#endif
