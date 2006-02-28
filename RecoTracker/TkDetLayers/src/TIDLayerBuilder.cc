#include "RecoTracker/TkDetLayers/interface/TIDLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TIDRingBuilder.h"

TIDLayer* TIDLayerBuilder::build(const GeometricDet* aTIDLayer,
				 const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricRings = aTIDLayer->components();
  //cout << "theGeometricRings.size(): " << theGeometricRings.size() << endl;

  TIDRingBuilder myBuilder;
  vector<const TIDRing*> theTIDRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){
    theTIDRings.push_back(myBuilder.build( *it,theGeomDetGeometry));    
  }

  return new TIDLayer(theTIDRings);
}
