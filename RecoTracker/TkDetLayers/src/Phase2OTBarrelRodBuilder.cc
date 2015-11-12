#include "Phase2OTBarrelRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2OTBarrelRod* Phase2OTBarrelRodBuilder::build(const GeometricDet* thePhase2OTBarrelRod,
						   const TrackerGeometry* theGeomDetGeometry)
{  
  vector<const GeometricDet*> allGeometricDets = thePhase2OTBarrelRod->components();
  LogDebug("TkDetLayers") << "Phase2OTBarrelRodBuilder with #Modules: " << allGeometricDets.size() << std::endl;

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  double meanR = 0;
  for(vector<const GeometricDet*>::const_iterator it=allGeometricDets.begin();
      it!=allGeometricDets.end();it++){
    meanR = meanR + (*it)->positionBounds().perp();
  }
  meanR = meanR/allGeometricDets.size();

  for(vector<const GeometricDet*>::iterator it=allGeometricDets.begin();
      it!=allGeometricDets.end(); it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );

    if( (*it)->positionBounds().perp() < meanR)
      innerGeomDets.push_back(theGeomDet);

    if( (*it)->positionBounds().perp() > meanR)
      outerGeomDets.push_back(theGeomDet);

  }

  LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size() ;
  LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size() ;

  return new Phase2OTBarrelRod(innerGeomDets,outerGeomDets);
 
}
