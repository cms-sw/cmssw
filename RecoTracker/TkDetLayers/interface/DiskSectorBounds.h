#ifndef RecoTracker_TkDetLayers_DiskSectorBounds_h
#define RecoTracker_TkDetLayers_DiskSectorBounds_h
 
 
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Surface/interface/Bounds.h"
#include <algorithm>
#include <cmath>

using namespace std;

class DiskSectorBounds : public Bounds {
public:
  
   DiskSectorBounds( float rmin, float rmax, float zmin, float zmax, float phiExt) : 
     theRmin(rmin), theRmax(rmax), theZmin(zmin), theZmax(zmax), thePhiExt(phiExt) {
     if ( theRmin > theRmax) swap( theRmin, theRmax);
     if ( theZmin > theZmax) swap( theZmin, theZmax);
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
 
 #endif 
 
