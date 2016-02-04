#include "DiskSectorBounds.h"

using namespace std;

bool DiskSectorBounds::inside( const Local3DPoint& p) const {
  
  // transform to system with local frame situated at disk center
  // and rotated  x/y axis
  Local3DPoint tmp( p.y()+theOffset, -p.x(), p.z());
  
  return  fabs(tmp.phi()) <= thePhiExt/2. &&
    tmp.perp() >= theRmin && tmp.perp() <= theRmax &&
    tmp.z() >= theZmin && tmp.z() <= theZmax ;

}

bool DiskSectorBounds::inside( const Local3DPoint& p, const LocalError& err, float scale) const {

  if (p.z() < theZmin || p.z() > theZmax) return false;

  Local3DPoint tmp( p.x(), p.y()+ theOffset, p.z());
  double perp2 = tmp.perp2();
  double perp = sqrt(perp2);

  // this is not really correct, should consider also the angle of the error ellipse
   if (perp2 == 0) return scale*scale*err.yy()   > theRmin*theRmin;

   LocalError tmpErr( err.xx(), err.xy(), err.yy());
   LocalError rotatedErr = tmpErr.rotate(tmp.x(), tmp.y());
   // x direction in this system is now r, phi is y
 
   float deltaR = scale*sqrt(rotatedErr.xx());
   float deltaPhi = atan( scale*sqrt(rotatedErr.yy())/perp);

   float tmpPhi = acos( tmp.y() / perp);
   
   return  perp >= max(theRmin-deltaR, 0.f) && perp <= theRmax+deltaR 
     && tmpPhi <= thePhiExt + deltaPhi ;  

}
