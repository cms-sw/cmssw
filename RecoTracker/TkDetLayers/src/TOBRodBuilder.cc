#include "TOBRodBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

TOBRod* TOBRodBuilder::build(GeometricDetPtr negTOBRod,
			     GeometricDetPtr posTOBRod,
			     const TrackerGeometry* theGeomDetGeometry)
{  
  vector<GeometricDetPtr>  theNegativeGeometricDets;
  if (negTOBRod != 0) theNegativeGeometricDets = negTOBRod->components();
  vector<GeometricDetPtr>  thePositiveGeometricDets;
  if (posTOBRod != 0) thePositiveGeometricDets = posTOBRod->components();

  auto allGeometricDets = theNegativeGeometricDets;
  allGeometricDets.insert(allGeometricDets.end(),thePositiveGeometricDets.begin(),
			  thePositiveGeometricDets.end());

  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  double meanR = (allGeometricDets[0]->positionBounds().perp()+allGeometricDets[1]->positionBounds().perp())/2;
  for(auto it=allGeometricDets.begin();
      it!=allGeometricDets.end(); it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );

    if( (*it)->positionBounds().perp() < meanR) 
      innerGeomDets.push_back(theGeomDet);
    
    if( (*it)->positionBounds().perp() > meanR) 
      outerGeomDets.push_back(theGeomDet);
    
  }
  
  //LogDebug("TkDetLayers") << "innerGeomDets.size(): " << innerGeomDets.size() ;
  //LogDebug("TkDetLayers") << "outerGeomDets.size(): " << outerGeomDets.size() ;
  return new TOBRod(innerGeomDets,outerGeomDets);
}
