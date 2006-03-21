#ifndef HitZCheck_H
#define HitZCheck_H

/** provides allowed range of Z coordinate from HitRZConstraint 
    at a given radius R */ 

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"


class HitZCheck : public HitRZCompatibility {
public:

  typedef TkTrackingRegionsMargin<float> Margin;

  HitZCheck() { }
  HitZCheck(const HitRZConstraint & rz, Margin margin = Margin(0,0))
    : theRZ(rz), theTolerance(margin) { }

  virtual bool operator() (const float & r, const float & z) const
    { return range(r).inside(z); }

  virtual Range range(const float & radius) const;

  virtual HitZCheck * clone() const { return new HitZCheck(*this); }

  void setTolerance(const Margin & tolerance) { theTolerance = tolerance; }

private:
  HitRZConstraint theRZ;
  Margin theTolerance;
};
#endif
