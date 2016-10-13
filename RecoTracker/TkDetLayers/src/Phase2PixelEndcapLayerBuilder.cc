#include "Phase2PixelEndcapLayerBuilder.h"
#include "Phase2OTEndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2PixelEndcapLayer* Phase2PixelEndcapLayerBuilder::build(const GeometricDet* aPhase2PixelEndcapLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  LogTrace("TkDetLayers") << "Phase2PixelEndcapLayerBuilder::build";
  vector<const GeometricDet*>  theGeometricRings = aPhase2PixelEndcapLayer->components();
  edm::LogInfo("TkDetLayers") << "theGeometricRings.size(): " << theGeometricRings.size() ;

  Phase2OTEndcapRingBuilder myBuilder;
  vector<const Phase2OTEndcapRing*> thePhase2OTEndcapRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){
    thePhase2OTEndcapRings.push_back(myBuilder.build( *it,theGeomDetGeometry,false ));
  }

  return new Phase2PixelEndcapLayer(thePhase2OTEndcapRings);
}
