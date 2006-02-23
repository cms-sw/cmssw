#ifndef TkDetLayers_PixelRodBuilder_h
#define TkDetLayers_PixelRodBuilder_h


#include "RecoTracker/TkDetLayers/interface/PixelRod.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"

/** A concrete builder for PixelRod 
 */
using namespace edm;
using namespace std;

class PixelRodBuilder {  
 public:
  PixelRodBuilder(){};
  PixelRod* build(const GeometricDet* aPixelRod,
	      const TrackingGeometry* theGeomDetGeometry);

  
};


#endif 
