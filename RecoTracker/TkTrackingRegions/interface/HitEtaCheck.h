#ifndef HitEtaCheck_H
#define HitEtaCheck_H

/** Fast Implementation of HitRZCompatibility.
    The RZConstraint is defined by two lines and their crossing point.
    The r-z compatibility is tested by comparistion of 
    cotangent given by r-z and lines crossing point with cotangents
    of two lines. */

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"

class HitEtaCheck GCC11_FINAL : public HitRZCompatibility {
public:

  static constexpr Algo me = etaAlgo;

  HitEtaCheck(bool inbarrel, 
      const HitRZConstraint::Point & point, 
      float cotLeftLine, float cotRightLine) 
    :  HitRZCompatibility(me), 
    isBarrel(inbarrel), 
    theRZ(HitRZConstraint(point, cotLeftLine, point, cotRightLine)) { }

  virtual bool operator() (const float & r, const float & z) const {
    const auto & lineLeft = theRZ.lineLeft();
    const auto & lineRight = theRZ.lineRight();
    float cotHit = (lineLeft.origin().z()-z)/(lineLeft.origin().r()-r);
    return lineRight.cotLine() < cotHit && cotHit < lineLeft.cotLine();
  }

  virtual Range range(const float & rORz) const {
    return (isBarrel) ? 
        HitZCheck(theRZ).range(rORz) : HitRCheck(theRZ).range(rORz);
  }
  virtual HitEtaCheck* clone() const { return new HitEtaCheck(*this); }
private:
  bool isBarrel;
  HitRZConstraint theRZ;
};
#endif
