#include "Phase2OTEndcapLayerBuilder.h"
#include "Phase2OTEndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTEndcapLayer* Phase2OTEndcapLayerBuilder::build(const GeometricDet* aPhase2OTEndcapLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricRings = aPhase2OTEndcapLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricRings.size(): " << theGeometricRings.size() ;

  Phase2OTEndcapRingBuilder myBuilder;
  vector<const Phase2OTEndcapRing*> thePhase2OTEndcapRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){
    thePhase2OTEndcapRings.push_back(myBuilder.build( *it,theGeomDetGeometry));    
  }

  return new Phase2OTEndcapLayer(thePhase2OTEndcapRings);
}
