#ifndef HitRZConstraint_H
#define HitRZConstraint_H

/** r-z constraint is formed by two PixelRecoLineRZ lines. */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

class HitRZConstraint {
public:

  typedef TkTrackingRegionsMargin<float> Margin;
  typedef PixelRecoRange<float> Range;
  typedef PixelRecoLineRZ::LineOrigin LineOrigin;

  HitRZConstraint() { }
  HitRZConstraint(
      const LineOrigin & leftPoint, float cotLeftLine,
      const LineOrigin & rightPoint, float cotRightLine)
    : theLineLeft(PixelRecoLineRZ(leftPoint, cotLeftLine)),
      theLineRight(PixelRecoLineRZ(rightPoint, cotRightLine)) { }
  HitRZConstraint(
      const PixelRecoLineRZ & lineLeft, 
      const PixelRecoLineRZ & lineRight)
    :  theLineLeft(lineLeft), theLineRight(lineRight) { } 

  const PixelRecoLineRZ & lineLeft() const { return theLineLeft; } 
  const PixelRecoLineRZ & lineRight() const { return theLineRight; } 

protected:
  PixelRecoLineRZ theLineLeft, theLineRight;
};

#endif
