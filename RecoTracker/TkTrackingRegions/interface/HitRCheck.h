#ifndef HitRCheck_H
#define HitRCheck_H

/** provides allowed range of radius R from HitRZConstraint 
    at a given Z coordinate */

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"

class HitRCheck GCC11_FINAL : public HitRZCompatibility {
public:
  static constexpr Algo me =rAlgo;

  typedef TkTrackingRegionsMargin<float> Margin;

  HitRCheck()  : HitRZCompatibility(me) { }
  HitRCheck(const HitRZConstraint & rz, Margin margin = Margin(0,0)) 
    :  HitRZCompatibility(me), theRZ(rz), theTolerance(margin) { } 

  virtual bool operator() (const float & r, const float & z) const
    { return range(z).inside(r); }

  inline Range range(const float & z) const;

  virtual HitRCheck * clone() const { return new HitRCheck(*this); }

  void setTolerance(const Margin & tolerance) { theTolerance = tolerance; }

private:
  HitRZConstraint theRZ;
  Margin theTolerance;
};



HitRCheck::Range HitRCheck::range(const float & z) const
{
  const float rBig = 150.; //something above the detector ranges
  const PixelRecoLineRZ & lineLeft = theRZ.lineLeft();
  const PixelRecoLineRZ & lineRight = theRZ.lineRight();

  if (z > 0.) {
    if (lineRight.cotLine() <= 0.) return Range(rBig, 0); //empty
    float rMin = lineRight.rAtZ(z);
    if (lineLeft.cotLine() <= 0) return Range(rMin-theTolerance.left(),rBig);
    float rMax = lineLeft.rAtZ(z);
    return Range(rMin-theTolerance.left(),rMax+theTolerance.right()); 
  } else {
    if (lineLeft.cotLine() >= 0.) return Range(rBig, 0); //empty 
    float rMin = lineLeft.rAtZ(z);
    if (lineRight.cotLine()>= 0) return Range(rMin-theTolerance.left(),rBig);
    float rMax = lineRight.rAtZ(z);
    return Range(rMin-theTolerance.left(),rMax+theTolerance.right());
  }
}
#endif
