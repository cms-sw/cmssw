#ifndef TkHitPairsCachedHit_H
#define TkHitPairsCachedHit_H

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
class TkHitPairsCachedHit {
public:
  TkHitPairsCachedHit( const SiPixelRecHit & hit) : theRecHit(hit) {
 
    //da cambiare 
    //   GlobalPoint gp = hit.globalPosition();
    LocalPoint gp = hit.localPosition();
    thePhi = gp.phi(); 
    theR = gp.perp();
    theZ = gp.z();
    //MP
    //  theRZ  = (hit.layer()->part()==barrel) ? theZ : theR;
    theRZ  = theZ ;
  }
  float phi() const {return thePhi;}
  float rOrZ() const { return theRZ; } 
  float r() const {return theR; }
  float z() const {return theZ; }

  SiPixelRecHit  RecHit() const { return theRecHit;}
  //  operator RecHit() const { return theRecHit;}
private:
  SiPixelRecHit theRecHit;
  float thePhi;
  float theR, theZ;
  float theRZ;
};

#endif 
