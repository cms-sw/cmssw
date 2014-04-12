#ifndef PixelRecoLineRZ_H
#define PixelRecoLineRZ_H

/** two dimensional line in r-z coordinates. line is defined by the point 
    and cotangent */

#include <cmath>

#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class PixelRecoLineRZ {
public:

  typedef PixelRecoPointRZ LineOrigin;

  PixelRecoLineRZ() { }

  PixelRecoLineRZ(const GlobalPoint & p1, const GlobalPoint & p2) 
    : theTIP2 ( initTIP2( p1.x(), p1.y(), p2.x(), p2.y() ) ),
      theOrigin ( LineOrigin( subTIP(p1.perp()), p1.z() ) ),
      theCotLine ( initCot( p2.z()-theOrigin.z(), subTIP(p2.perp())-theOrigin.r() ) )
    { }

  PixelRecoLineRZ(const LineOrigin & aOrigin, float aCotLine, float transverseIP = 0.f)
    : theTIP2( transverseIP*transverseIP ),
      theOrigin( subTIP(aOrigin.r()), aOrigin.z() ),
      theCotLine(aCotLine) { }

  PixelRecoLineRZ(const LineOrigin & aOrigin, const PixelRecoPointRZ & aPoint, float transverseIP = 0.f)
    : theTIP2( transverseIP*transverseIP ),
      theOrigin( subTIP(aOrigin.r()), aOrigin.z() ),
      theCotLine( initCot( aPoint.z()-theOrigin.z(), subTIP(aPoint.r())-theOrigin.r() ) )
    { } 

  float cotLine() const { return theCotLine; }
  float transverseIP() const { return std::sqrt(theTIP2); }
  float transverseIP2() const { return theTIP2; }
  LineOrigin origin() const { return LineOrigin( addTIP(theOrigin.r()), theOrigin.z() ); }

  float zAtR (float r) const 
    { return theOrigin.z()+(subTIP(r)-theOrigin.r())*theCotLine; }
  float rAtZ (float z) const 
    { return (std::abs(theCotLine) > 1.e-4f) ? addTIP(theOrigin.r()+(z-theOrigin.z())/theCotLine) : 99999.f; }

private:
  static float initCot (float dz, float dr)
    { return (std::abs(dr) > 1.e-4f)  ? dz/dr : 99999.f; }
  static float initTIP2 (float x1, float y1, float x2, float y2)
    {
      double l = y1 * (x2 - x1) - x1 * (y2 - y1);
      return l * l / ( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) );
    }

  inline float addTIP (float val) const
    { return theTIP2 ? std::sqrt(val * val + theTIP2) : val; }
  inline float subTIP (float val) const
    {
      if (!theTIP2) return val;
      float val2 = val * val;
      return val2 > theTIP2 ? std::sqrt(val2 - theTIP2) : 0.;
    }

private:
  float theTIP2;
  LineOrigin theOrigin;
  float theCotLine;
};

// simpler version (no tip)
class SimpleLineRZ {
public:
  
  typedef PixelRecoPointRZ Point;
  
  SimpleLineRZ() { }
  
  SimpleLineRZ(const Point & aOrigin, float aCotLine) :
    theOrigin( aOrigin ),
    theCotLine(aCotLine) { }
  
  SimpleLineRZ(const Point & aOrigin, const Point & aPoint) :
    theOrigin( aOrigin ),
    theCotLine( (aPoint.z()-theOrigin.z())/ (aPoint.r()-theOrigin.r()) )
  { } 
  
  float cotLine() const { return theCotLine; }
  Point const & origin() const { return theOrigin; }
  
  float zAtR (float r) const { return theOrigin.z()+(r-theOrigin.r())*theCotLine; }
  float rAtZ (float z) const { return theOrigin.r()+(z-theOrigin.z())/theCotLine; }
  
  
private:
  Point theOrigin;
  float theCotLine=0;
};

#endif
