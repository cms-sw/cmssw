#include "Phase2OTBarrelRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

Phase2OTBarrelRod* Phase2OTBarrelRodBuilder::build(const GeometricDet* thePhase2OTBarrelRod,
							   const TrackerGeometry* theGeomDetGeometry)
{  
  vector<const GeometricDet*> allGeometricDets = thePhase2OTBarrelRod->components();

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;
  vector<const GeomDet*> innerGeomDetBrothers;
  vector<const GeomDet*> outerGeomDetBrothers;

  // compute meanR using the first and the third module because of the pt module pairs
  LogDebug("Phase2OTBarrelRodRadii") << "mean computed with " 
				     << allGeometricDets[0]->positionBounds().perp() 
				     << " and " << allGeometricDets[2]->positionBounds().perp() 
				     << " and " << allGeometricDets[1]->positionBounds().perp() 
				     << " and " << allGeometricDets[3]->positionBounds().perp() ;
  double meanR = (allGeometricDets[0]->positionBounds().perp()+allGeometricDets[2]->positionBounds().perp())/2;
  double meanRBrothers = (allGeometricDets[1]->positionBounds().perp()+allGeometricDets[3]->positionBounds().perp())/2;

  unsigned int counter=0;
  for(vector<const GeometricDet*>::iterator it=allGeometricDets.begin();
      it!=allGeometricDets.end(); it++,counter++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );

    if(counter%2==0) {
      if( (*it)->positionBounds().perp() < meanR) 
	innerGeomDets.push_back(theGeomDet);
    
      if( (*it)->positionBounds().perp() > meanR) 
	outerGeomDets.push_back(theGeomDet);
    }
    else {
      if( (*it)->positionBounds().perp() < meanRBrothers) 
	innerGeomDetBrothers.push_back(theGeomDet);
    
      if( (*it)->positionBounds().perp() > meanRBrothers) 
	outerGeomDetBrothers.push_back(theGeomDet);
    }
  }
  
  //LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size() ;
  //LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size() ;
  return new Phase2OTBarrelRod(innerGeomDets,outerGeomDets,innerGeomDetBrothers,outerGeomDetBrothers);
}
