#ifndef TkNavigation_TkLayerLess_H
#define TkNavigation_TkLayerLess_H

#include "TrackingTools/DetLayers/interface/NavigationDirection.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include <functional>

/** Defines order of layers in the Tracker as seen by straight tracks
 *  coming from the interaction region.
 */

class TkLayerLess 
  : public std::binary_function< const DetLayer*,const DetLayer*,bool> {
public:

    TkLayerLess( NavigationDirection dir = insideOut, const DetLayer * fromLayer = 0) :
    theDir(dir) {
    if (fromLayer){
      theOriginLayer = true;
      theFromLayerSign = (fromLayer->position().z()>0 ? 1 : -1) ;
    }else theOriginLayer = false;
  }

  bool operator()( const DetLayer* a, const DetLayer* b) const {
    if (!theOriginLayer){
      if (theDir == insideOut) return insideOutLess( a, b);
      else return insideOutLess( b, a);
    }
    else{
      if (theDir == insideOut) return insideOutLessSigned( a, b);
      else return insideOutLessSigned(b, a);
    }
  }

private:

  NavigationDirection theDir;
  bool theOriginLayer; //true take into account next parameter, false, do as usual
  int theFromLayerSign; //1 z>0: -1 z<0

  bool insideOutLess( const DetLayer*,const DetLayer*) const;
  bool insideOutLessSigned( const DetLayer*,const DetLayer*) const;

  bool barrelForwardLess( const BarrelDetLayer* blb,
			  const ForwardDetLayer* fla) const;

};

#endif
