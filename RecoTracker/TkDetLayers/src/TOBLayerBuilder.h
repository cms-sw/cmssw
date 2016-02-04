#ifndef TkDetLayers_TOBLayerBuilder_h
#define TkDetLayers_TOBLayerBuilder_h


#include "TOBLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TOBLayer 
 */

#pragma GCC visibility push(hidden)
class TOBLayerBuilder {  
 public:
  TOBLayerBuilder(){};
  TOBLayer* build(const GeometricDet* aTOBLayer,
		  const TrackerGeometry* theGeomDetGeometry);

  
};


#pragma GCC visibility pop
#endif 
