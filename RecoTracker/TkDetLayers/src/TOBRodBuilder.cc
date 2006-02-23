#include "RecoTracker/TkDetLayers/interface/TOBRodBuilder.h"

TOBRod* TOBRodBuilder::build(const GeometricDet* negTOBRod,
			     const GeometricDet* posTOBRod,
			     const TrackingGeometry* theGeomDetGeometry){
  
  vector<const GeometricDet*>  theNegativeGeometricDets = negTOBRod->components();
  vector<const GeometricDet*>  thePositiveGeometricDets = posTOBRod->components();

  vector<const GeometricDet*> allGeometricDets = theNegativeGeometricDets;
  allGeometricDets.insert(allGeometricDets.end(),thePositiveGeometricDets.begin(),
			  thePositiveGeometricDets.end());

  //cout << "allGeometricDets.size():      " << allGeometricDets.size()         << endl;
  
  vector<const GeomDet*> innerGeomDets;
  vector<const GeomDet*> outerGeomDets;

  double meanR = (allGeometricDets[0]->positionBounds().perp()+allGeometricDets[1]->positionBounds().perp())/2;
  for(vector<const GeometricDet*>::iterator it=allGeometricDets.begin();
      it!=allGeometricDets.end(); it++){
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet( (*it)->geographicalID() );

    if( (*it)->positionBounds().perp() < meanR) 
      innerGeomDets.push_back(theGeomDet);
    
    if( (*it)->positionBounds().perp() > meanR) 
      outerGeomDets.push_back(theGeomDet);
    
  }
  
  //cout << "innerGeomDets.size(): " << innerGeomDets.size() << endl;
  //cout << "outerGeomDets.size(): " << outerGeomDets.size() << endl;
  return new TOBRod(innerGeomDets,outerGeomDets);
}
