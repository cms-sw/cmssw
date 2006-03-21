
#include <cmath>
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"

HitZCheck::Range HitZCheck::range(const float & radius) const
{
  return Range( theRZ.lineLeft().zAtR(radius) - theTolerance.left(), 
                theRZ.lineRight().zAtR(radius) + theTolerance.right());
}
