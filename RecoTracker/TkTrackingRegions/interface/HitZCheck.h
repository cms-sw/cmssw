#ifndef HitZCheck_H
#define HitZCheck_H

/** provides allowed range of Z coordinate from HitRZConstraint 
    at a given radius R */ 

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"


class HitZCheck GCC11_FINAL : public HitRZCompatibility {
public:
  static constexpr Algo me =zAlgo;

  typedef TkTrackingRegionsMargin<float> Margin;

  HitZCheck()  : HitRZCompatibility(me) { }
  HitZCheck(const HitRZConstraint & rz, Margin margin = Margin(0,0))
    : HitRZCompatibility(me), theRZ(rz), theTolerance(margin) { }

  virtual bool operator() (const float & r, const float & z) const
    { return range(r).inside(z); }

  inline Range range(const float & radius) const;

  virtual HitZCheck * clone() const { return new HitZCheck(*this); }

  void setTolerance(const Margin & tolerance) { theTolerance = tolerance; }

private:
  HitRZConstraint theRZ;
  Margin theTolerance;
};

HitZCheck::Range HitZCheck::range(const float & radius) const
{
  return Range( theRZ.lineLeft().zAtR(radius) - theTolerance.left(), 
                theRZ.lineRight().zAtR(radius) + theTolerance.right());
}


#endif
