#ifndef RecoTracker_TkDetLayers_DiskSectorBounds_h
#define RecoTracker_TkDetLayers_DiskSectorBounds_h
 
 
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <algorithm>
#include <cmath>
#include <cassert>

#pragma GCC visibility push(hidden)
class DiskSectorBounds GCC11_FINAL : public Bounds {
public:
  
   DiskSectorBounds( float rmin, float rmax, float zmin, float zmax, float phiExt) : 
     theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax), thePhiExtH(0.5f*phiExt) {
     assert(thePhiExtH>0);
     if ( theRmin > theRmax) std::swap( theRmin, theRmax);
     if ( theZmin > theZmax) std::swap( theZmin, theZmax);
     theOffset = theRmin +  0.5f*(theRmax-theRmin);
   }
   
   virtual float length()    const { return theRmax-theRmin*std::cos(thePhiExtH);}
   virtual float width()     const { return 2.f*theRmax*std::sin(thePhiExtH);}
   virtual float thickness() const { return theZmax-theZmin;}
 
   
   
   virtual bool inside( const Local3DPoint& p) const;
     
   virtual bool inside( const Local3DPoint& p, const LocalError& err, float scale) const;
 
   virtual Bounds* clone() const { 
     return new DiskSectorBounds(*this);
   }
 
   float innerRadius() const {return theRmin;}
   float outerRadius() const {return theRmax;}
   float phiHalfExtension() const {return thePhiExtH;}
 
 private:
   float theRmin;
   float theRmax;
   float theZmin;
   float theZmax;
   float thePhiExtH;
   float theOffset;
 };
 
#pragma GCC visibility pop
#endif 
 
