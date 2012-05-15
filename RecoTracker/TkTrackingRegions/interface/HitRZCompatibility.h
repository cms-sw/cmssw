#ifndef HitRZCompatibility_H
#define HitRZCompatibility_H

/** abstract class to check if r-z coordinates or comptible with the region */
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

class HitRZCompatibility {
public:
  virtual ~HitRZCompatibility() {}
  typedef PixelRecoRange<float> Range;
  virtual bool operator() (const float & r, const float & z) const = 0;
  virtual Range range(const float & rORz) const = 0; 
  virtual HitRZCompatibility * clone() const = 0;
};
#endif
