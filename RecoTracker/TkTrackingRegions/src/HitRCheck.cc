#include <cmath>
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"

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
