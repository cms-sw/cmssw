#ifndef RecoTracker_TkDetLayers_DiskSectorBounds_h
#define RecoTracker_TkDetLayers_DiskSectorBounds_h
 
 
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include <algorithm>
#include <cmath>


#pragma GCC visibility push(hidden)
class DiskSectorBounds GCC11_FINAL : public Bounds {
public:
  
   DiskSectorBounds( float rmin, float rmax, float zmin, float zmax, float phiExt) : 
     theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax), thePhiExt(phiExt) {
     if ( theRmin > theRmax) std::swap( theRmin, theRmax);
     if ( theZmin > theZmax) std::swap( theZmin, theZmax);
     theOffset = theRmin + (theRmax-theRmin)/2. ;
   }
   
   virtual float length()    const { return theRmax-theRmin*cos(thePhiExt/2.);}
   virtual float width()     const { return 2*theRmax*sin(thePhiExt/2.);}
   virtual float thickness() const { return theZmax-theZmin;}
 
   virtual bool inside( const Local3DPoint& p) const;
     
   virtual bool inside( const Local3DPoint& p, const LocalError& err, float scale) const;
 
   virtual bool inside( const Local2DPoint& p, const LocalError& err) const {
     return Bounds::inside(p,err);
   }
 
   virtual Bounds* clone() const { 
     return new DiskSectorBounds(*this);
   }
 
   float innerRadius() const {return theRmin;}
   float outerRadius() const {return theRmax;}
   float phiExtension() const {return thePhiExt;}
 
 private:
   float theRmin;
   float theRmax;
   float theZmin;
   float theZmax;
   float thePhiExt;
   float theOffset;
 };
 
#pragma GCC visibility pop
#endif 
 
