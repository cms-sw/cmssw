#include "Phase2OTEndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTEndcapRing* Phase2OTEndcapRingBuilder::build(const GeometricDet* aPhase2OTEndcapRing,
			 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricDets = aPhase2OTEndcapRing->components();
  LogDebug("TkDetLayers") << "Phase2OTEndcapRingBuilder with #Modules: " << theGeometricDets.size() << std::endl;

  vector<const GeomDet*> frontGeomDets;
  vector<const GeomDet*> backGeomDets;

  //---- to evaluate meanZ
  double meanZ = 0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++){
    meanZ = meanZ + (*it)->positionBounds().z();
  }
  meanZ = meanZ/theGeometricDets.size();
  //----

  for(vector<const GeometricDet*>::const_iterator it=theGeometricDets.begin();
      it!=theGeometricDets.end();it++){

    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );

    if( fabs( (*it)->positionBounds().z() ) < fabs(meanZ))
      frontGeomDets.push_back(theGeomDet);

    if( fabs( (*it)->positionBounds().z() ) > fabs(meanZ))
      backGeomDets.push_back(theGeomDet);      
  }

  LogDebug("TkDetLayers") << "frontGeomDets.size(): " << frontGeomDets.size() ;
  LogDebug("TkDetLayers") << "backGeomDets.size(): " << backGeomDets.size() ;

  return new Phase2OTEndcapRing(frontGeomDets,backGeomDets);

}
