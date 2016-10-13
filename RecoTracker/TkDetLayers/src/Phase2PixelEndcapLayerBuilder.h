#ifndef TkDetLayers_Phase2PixelEndcapLayerBuilder_h
#define TkDetLayers_Phase2PixelEndcapLayerBuilder_h


#include "Phase2PixelEndcapLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2PixelEndcapLayer 
 */

#pragma GCC visibility push(hidden)
class Phase2PixelEndcapLayerBuilder {  
 public:
  Phase2PixelEndcapLayerBuilder(){};
  Phase2PixelEndcapLayer* build(const GeometricDet* aPhase2PixelEndcapLayer,
  		             const TrackerGeometry* theGeomDetGeometry) __attribute__ ((cold));

   
};


#pragma GCC visibility pop
#endif 
