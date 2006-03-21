#ifndef HitRCheck_H
#define HitRCheck_H

/** provides allowed range of radius R from HitRZConstraint 
    at a given Z coordinate */

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"

class HitRCheck: public HitRZCompatibility {
public:

  typedef TkTrackingRegionsMargin<float> Margin;

  HitRCheck() { }
  HitRCheck(const HitRZConstraint & rz, Margin margin = Margin(0,0)) 
    : theRZ(rz), theTolerance(margin) { } 

  virtual bool operator() (const float & r, const float & z) const
    { return range(z).inside(r); }

  virtual Range range(const float & z) const;

  virtual HitRCheck * clone() const { return new HitRCheck(*this); }

  void setTolerance(const Margin & tolerance) { theTolerance = tolerance; }

private:
  HitRZConstraint theRZ;
  Margin theTolerance;
};
#endif
