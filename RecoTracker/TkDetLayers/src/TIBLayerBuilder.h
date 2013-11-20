#ifndef TkDetLayers_TIBLayerBuilder_h
#define TkDetLayers_TIBLayerBuilder_h


#include "TIBLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for TIBLayer 
 */

#pragma GCC visibility push(hidden)
class TIBLayerBuilder {  
 public:
  TIBLayerBuilder(){};
  TIBLayer* build(GeometricDetPtr aTIBLayer,
		  const TrackerGeometry* theGeomDetGeometry);

  void constructRings(std::vector<GeometricDetPtr>& theGeometricRods,
		      std::vector<std::vector<GeometricDetPtr> >& innerGeometricDetRings,
		      std::vector<std::vector<GeometricDetPtr> >& outerGeometricDetRings);
  
};


#pragma GCC visibility pop
#endif 
