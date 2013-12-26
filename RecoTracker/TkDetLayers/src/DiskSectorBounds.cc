#include "DiskSectorBounds.h"

using namespace std;

bool DiskSectorBounds::inside( const Local3DPoint& p) const {
  
  // transform to system with local frame situated at disk center
  // and rotated  x/y axis
  Local3DPoint tmp( p.y()+theOffset, -p.x(), p.z());
  
  return  std::abs(tmp.barePhi()) <= 0.5f*thePhiExt &&
    tmp.perp() >= theRmin && tmp.perp() <= theRmax &&
    tmp.z() >= theZmin && tmp.z() <= theZmax ;

}

bool DiskSectorBounds::inside( const Local3DPoint& p, const LocalError& err, float scale) const {

  if ( (p.z() < theZmin) | (p.z() > theZmax)) return false;

  Local3DPoint tmp( p.x(), p.y()+ theOffset, p.z());
  auto perp2 = tmp.perp2();
  auto perp = std::sqrt(perp2);

  // this is not really correct, should consider also the angle of the error ellipse
   if (perp2 == 0) return scale*scale*err.yy()   > theRmin*theRmin;

   LocalError tmpErr( err.xx(), err.xy(), err.yy());
   LocalError rotatedErr = tmpErr.rotate(tmp.x(), tmp.y());
   // x direction in this system is now r, phi is y
 
   float deltaR = scale*std::sqrt(rotatedErr.xx());
   float deltaPhi = std::atan( scale*std::sqrt(rotatedErr.yy())/perp);

   float tmpPhi = std::acos( tmp.y() / perp);
   
   return  (perp >= std::max(theRmin-deltaR, 0.f)) & (perp <= theRmax+deltaR) 
     & (tmpPhi <= thePhiExt + deltaPhi) ;  

}
