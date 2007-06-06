#ifndef PixelRecoLineRZ_H
#define PixelRecoLineRZ_H

/** two dimensional line in r-z coordinates. line is defined by the point 
    and cotangent */

#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class PixelRecoLineRZ {
public:

  typedef PixelRecoPointRZ LineOrigin;

  PixelRecoLineRZ() { }

  PixelRecoLineRZ(const GlobalPoint & p1, const GlobalPoint & p2) 
    : theOrigin ( LineOrigin( p1.perp(), p1.z()) ),
      theCotLine ( initCot(theOrigin.z()-p2.z(), theOrigin.r()-p2.perp()) ) 
  { }


  PixelRecoLineRZ(const LineOrigin & aOrigin, float aCotLine)
    : theOrigin(aOrigin), theCotLine(aCotLine) { }

  PixelRecoLineRZ(const LineOrigin & aOrigin, const PixelRecoPointRZ & aPoint) 
    : theOrigin(aOrigin),
      theCotLine( initCot( aPoint.z()-aOrigin.z(), aPoint.r()-aOrigin.r() ) ) 
      { } 

  float  cotLine() const {return theCotLine; }
  const LineOrigin & origin() const  { return theOrigin; }

  float zAtR (float r) const 
    { return theOrigin.z()+(r-theOrigin.r())*theCotLine; }
  float rAtZ (float z) const 
    { return (fabs(theCotLine) > 1.e-4) ? theOrigin.r()+(z-theOrigin.z())/theCotLine : 99999.; }

private:
  float initCot (float dz, float dr) {
    return (fabs(dr) > 1.e-4)  ? dz/dr : 99999.;
  }
private:
  LineOrigin theOrigin;
  float theCotLine;
};
#endif
