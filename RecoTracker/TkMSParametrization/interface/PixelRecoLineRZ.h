#ifndef PixelRecoLineRZ_H
#define PixelRecoLineRZ_H

/** two dimensional line in r-z coordinates. line is defined by the point 
    and cotangent */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"

class PixelRecoLineRZ {
public:

  typedef PixelRecoPointRZ LineOrigin;

  PixelRecoLineRZ() { }
  PixelRecoLineRZ(const LineOrigin & aOrigin, float aCotLine)
    : theOrigin(aOrigin), theCotLine(aCotLine) { }
  PixelRecoLineRZ(const LineOrigin & aOrigin, 
                          const PixelRecoPointRZ & aPoint) 
    : theOrigin(aOrigin),
      theCotLine( (aPoint.z()-aOrigin.z())/(aPoint.r()-aOrigin.r()) )
      { } 

  float  cotLine() const {return theCotLine; }
  const LineOrigin & origin() const  { return theOrigin; }
  

  float zAtR (float r) const 
    { return theOrigin.z()+(r-theOrigin.r())*theCotLine; }
  float rAtZ (float z) const 
    { return theOrigin.r()+(z-theOrigin.z())/theCotLine; }

private:
  LineOrigin theOrigin;
  float theCotLine;
};
#endif
