#ifndef HitRZConstraint_H
#define HitRZConstraint_H

/** r-z constraint is formed by two PixelRecoLineRZ lines. */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

class HitRZConstraint {
public:
  using Line = SimpleLineRZ;
  using Point = SimpleLineRZ::Point;

  HitRZConstraint() { }
  HitRZConstraint(
      const Point & leftPoint, float cotLeftLine,
      const Point & rightPoint, float cotRightLine)
    : theLineLeft(Line(leftPoint, cotLeftLine)),
      theLineRight(Line(rightPoint, cotRightLine)) { }
  HitRZConstraint(
      const Line & lineLeft, 
      const Line & lineRight)
    :  theLineLeft(lineLeft), theLineRight(lineRight) { } 

  const Line & lineLeft() const { return theLineLeft; } 
  const Line & lineRight() const { return theLineRight; } 

protected:
  Line theLineLeft, theLineRight;
};

#endif
