#include "Phase2OTEndcapLayerBuilder.h"
#include "Phase2OTEndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTEndcapLayer* Phase2OTEndcapLayerBuilder::build(const GeometricDet* aPhase2OTEndcapLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  LogTrace("TkDetLayers") << "Phase2ForwardLayerBuilder::build";
  vector<const GeometricDet*>  theGeometricRings = aPhase2OTEndcapLayer->components();
  edm::LogInfo("TkDetLayers") << "theGeometricRings.size(): " << theGeometricRings.size() ;

  Phase2OTEndcapRingBuilder myBuilder;
  vector<const Phase2OTEndcapRing*> thePhase2OTEndcapRings;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRings.begin();
      it!=theGeometricRings.end();it++){

    if( (*it)->type() == GeometricDet::OTPhase2Wheel){
      LogTrace("TkDetLayers") << "GeometricDet::OTPhase2Wheel";
      thePhase2OTEndcapRings.push_back(myBuilder.build( *it,theGeomDetGeometry,true ));
    } else if( (*it)->type() == GeometricDet::PixelPhase2FullDisk || (*it)->type() == GeometricDet::PixelPhase2ReducedDisk || (*it)->type() == GeometricDet::PixelPhase2TDRDisk){
      LogTrace("TkDetLayers") << "GeometricDet::PixelPhase2Disk";
    } else if( (*it)->type() == GeometricDet::panel){
      LogTrace("TkDetLayers") << "GeometricDet::panel";
      thePhase2OTEndcapRings.push_back(myBuilder.build( *it,theGeomDetGeometry,false ));
    } else {
      LogTrace("TkDetLayers") << "Not GeometricDet::OTPhase2Wheel or panel or some PixelPhase2Disks!!";
    }

  }

  return new Phase2OTEndcapLayer(thePhase2OTEndcapRings);
}
