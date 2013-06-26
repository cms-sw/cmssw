#ifndef HitRZCompatibility_H
#define HitRZCompatibility_H

/** abstract class to check if r-z coordinates or comptible with the region */
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class HitRZCompatibility {
public:
  // only three algos are implemented..
  enum Algo { zAlgo=0,rAlgo=1,etaAlgo=2};
public:
  typedef PixelRecoRange<float> Range;

  explicit HitRZCompatibility(Algo a) : m_algo(a){}
  virtual ~HitRZCompatibility() {}
  virtual bool operator() (const float & r, const float & z) const = 0;
  virtual Range range(const float & rORz) const = 0; 
  virtual HitRZCompatibility * clone() const = 0;
  Algo algo() const { return m_algo;}
  Algo m_algo;
};
#endif
