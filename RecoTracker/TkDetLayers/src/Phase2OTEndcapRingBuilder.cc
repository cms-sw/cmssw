#include "Phase2OTEndcapRingBuilder.h"

using namespace edm;
using namespace std;

Phase2OTEndcapRing* Phase2OTEndcapRingBuilder::build(const GeometricDet* aPhase2OTEndcapRing,
			 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  allGeometricDets = aPhase2OTEndcapRing->components();
  vector<const GeometricDet*>  compGeometricDets;
  LogDebug("TkDetLayers") << "Phase2OTEndcapRingBuilder with #Modules: " << allGeometricDets.size() << std::endl;

  vector<const GeomDet*> frontGeomDets;
  vector<const GeomDet*> backGeomDets;
  vector<const GeomDet*> frontGeomDetBrothers;
  vector<const GeomDet*> backGeomDetBrothers;

  //---- to evaluate meanZ
  double meanZ = 0;
  double meanZBrothers = 0;
  for(vector<const GeometricDet*>::const_iterator it=allGeometricDets.begin(); it!=allGeometricDets.end();it++){
    compGeometricDets = (*it)->components();
    if (compGeometricDets.size() != 2){
      LogDebug("TkDetLayers") << " Stack not with two components but with " << compGeometricDets.size() << std::endl;
    } else {
      LogTrace("TkDetLayers") << " compGeometricDets[0]->positionBounds().perp() " << compGeometricDets[0]->positionBounds().z() << std::endl;
      LogTrace("TkDetLayers") << " compGeometricDets[1]->positionBounds().perp() " << compGeometricDets[1]->positionBounds().z() << std::endl;
      meanZ = meanZ + compGeometricDets[0]->positionBounds().z();
      meanZBrothers = meanZBrothers + compGeometricDets[1]->positionBounds().z();
    }

  }
  meanZ = meanZ/allGeometricDets.size();
  meanZBrothers = meanZBrothers/allGeometricDets.size();
  LogDebug("TkDetLayers") << " meanZ Lower " << meanZ << std::endl;
  LogDebug("TkDetLayers") << " meanZ Upper " << meanZBrothers << std::endl;

  for(vector<const GeometricDet*>::const_iterator it=allGeometricDets.begin();
      it!=allGeometricDets.end();it++){
    compGeometricDets = (*it)->components(); 
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( compGeometricDets[0]->geographicalID() );
    LogTrace("TkDetLayers") << " inserisco " << compGeometricDets[0]->geographicalID().rawId() << std::endl;

    if( fabs( compGeometricDets[0]->positionBounds().z() ) < fabs(meanZ))
      frontGeomDets.push_back(theGeomDet);

    if( fabs( compGeometricDets[0]->positionBounds().z() ) > fabs(meanZ))
      backGeomDets.push_back(theGeomDet);      

    const GeomDet* theGeomDetBrother = theGeomDetGeometry->idToDet( compGeometricDets[1]->geographicalID() );
    LogTrace("TkDetLayers") << " inserisco " << compGeometricDets[1]->geographicalID().rawId() << std::endl;

    if( fabs( compGeometricDets[1]->positionBounds().z() ) < fabs(meanZ))
      frontGeomDetBrothers.push_back(theGeomDetBrother);

    if( fabs( compGeometricDets[1]->positionBounds().z() ) > fabs(meanZ))
      backGeomDetBrothers.push_back(theGeomDetBrother);
  }

  LogDebug("TkDetLayers") << "frontGeomDets.size(): " << frontGeomDets.size() ;
  LogDebug("TkDetLayers") << "backGeomDets.size(): " << backGeomDets.size() ;
  LogDebug("TkDetLayers") << "frontGeomDetsBro.size(): " << frontGeomDetBrothers.size() ;
  LogDebug("TkDetLayers") << "backGeomDetsBro.size(): " << backGeomDetBrothers.size() ;

  return new Phase2OTEndcapRing(frontGeomDets,backGeomDets,frontGeomDetBrothers,backGeomDetBrothers);

}
