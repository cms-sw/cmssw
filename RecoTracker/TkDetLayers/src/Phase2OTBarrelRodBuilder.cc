#include "Phase2OTBarrelRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2OTBarrelRod* Phase2OTBarrelRodBuilder::build(const GeometricDet* thePhase2OTBarrelRod,
						   const TrackerGeometry* theGeomDetGeometry)
{  
  vector<const GeometricDet*> allGeometricDets = thePhase2OTBarrelRod->components();
  vector<const GeometricDet*> compGeometricDets;
  LogDebug("TkDetLayers") << "Phase2OTBarrelRodBuilder with #Modules: " << allGeometricDets.size() << std::endl;

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;
  vector<const GeomDet*> innerGeomDetBrothers;
  vector<const GeomDet*> outerGeomDetBrothers;

  double meanR = 0;
  double meanRBrothers = 0;
  for(vector<const GeometricDet*>::const_iterator it=allGeometricDets.begin(); it!=allGeometricDets.end();it++){
    compGeometricDets = (*it)->components(); 
    if (compGeometricDets.size() != 2){
      LogDebug("TkDetLayers") << " Stack not with two components but with " << compGeometricDets.size() << std::endl;
    } else {
      //LogTrace("TkDetLayers") << " compGeometricDets[0]->positionBounds().perp() " << compGeometricDets[0]->positionBounds().perp() << std::endl;
      //LogTrace("TkDetLayers") << " compGeometricDets[1]->positionBounds().perp() " << compGeometricDets[1]->positionBounds().perp() << std::endl;
      meanR = meanR + compGeometricDets[0]->positionBounds().perp();
      meanRBrothers = meanRBrothers + compGeometricDets[1]->positionBounds().perp();
    }

  }
  meanR = meanR/allGeometricDets.size();
  meanRBrothers = meanRBrothers/allGeometricDets.size();
  LogDebug("TkDetLayers") << " meanR Lower " << meanR << std::endl;
  LogDebug("TkDetLayers") << " meanR Upper " << meanRBrothers << std::endl;

  for(vector<const GeometricDet*>::iterator it=allGeometricDets.begin(); it!=allGeometricDets.end(); it++){
    compGeometricDets = (*it)->components(); 
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( compGeometricDets[0]->geographicalID() );
    LogTrace("TkDetLayers") << " inserisco " << compGeometricDets[0]->geographicalID().rawId() << std::endl;

    if( compGeometricDets[0]->positionBounds().perp() < meanR)
      innerGeomDets.push_back(theGeomDet);

    if( compGeometricDets[0]->positionBounds().perp() > meanR)
      outerGeomDets.push_back(theGeomDet);

    const GeomDet* theGeomDetBrother = theGeomDetGeometry->idToDet( compGeometricDets[1]->geographicalID() );
    LogTrace("TkDetLayers") << " inserisco " << compGeometricDets[1]->geographicalID().rawId() << std::endl;
    if( compGeometricDets[1]->positionBounds().perp() < meanRBrothers) 
      innerGeomDetBrothers.push_back(theGeomDetBrother);
    
    if( compGeometricDets[1]->positionBounds().perp() > meanRBrothers) 
      outerGeomDetBrothers.push_back(theGeomDetBrother);
  }

  LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size() ;
  LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size() ;
  LogDebug("TkDetLayers") << "innerGeomDetsBro.size(): " << innerGeomDetBrothers.size() ;
  LogDebug("TkDetLayers") << "outerGeomDetsBro.size(): " << outerGeomDetBrothers.size() ;

  return new Phase2OTBarrelRod(innerGeomDets,outerGeomDets,innerGeomDetBrothers,outerGeomDetBrothers);
 
}
