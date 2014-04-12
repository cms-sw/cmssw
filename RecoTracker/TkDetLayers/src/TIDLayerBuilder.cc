#include "TIDLayerBuilder.h"
#include "TIDRingBuilder.h"

using namespace edm;
using namespace std;

TIDLayer* TIDLayerBuilder::build(const GeometricDet* aTIDLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricRings = aTIDLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricRings.size(): " << theGeometricRings.size() ;

  TIDRingBuilder myBuilder;
  vector<const TIDRing*> theTIDRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){
    theTIDRings.push_back(myBuilder.build( *it,theGeomDetGeometry));    
  }

  return new TIDLayer(theTIDRings);
}
