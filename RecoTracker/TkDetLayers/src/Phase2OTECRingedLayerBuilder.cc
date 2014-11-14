#include "Phase2OTECRingedLayerBuilder.h"
#include "Phase2OTECRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTECRingedLayer* Phase2OTECRingedLayerBuilder::build(const GeometricDet* aPhase2OTECRingedLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricRings = aPhase2OTECRingedLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricRings.size(): " << theGeometricRings.size() ;

  Phase2OTECRingBuilder myBuilder;
  vector<const Phase2OTECRing*> thePhase2OTECRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){
    thePhase2OTECRings.push_back(myBuilder.build( *it,theGeomDetGeometry));    
  }

  return new Phase2OTECRingedLayer(thePhase2OTECRings);
}
