#include "RecoTracker/TkDetLayers/interface/TOBRodBuilder.h"

using namespace std;

TOBRod* TOBRodBuilder::build(const GeometricDet* negTOBRod,
			     const GeometricDet* posTOBRod,
			     const TrackerGeometry* theGeomDetGeometry)
{  

  vector<const GeometricDet*>  theNegativeGeometricDets;
  if (negTOBRod != 0) theNegativeGeometricDets = negTOBRod->components();
  vector<const GeometricDet*>  thePositiveGeometricDets;
  if (posTOBRod != 0) thePositiveGeometricDets = posTOBRod->components();

  vector<const GeometricDet*> allGeometricDets = theNegativeGeometricDets;
  allGeometricDets.insert(allGeometricDets.end(),thePositiveGeometricDets.begin(),
			  thePositiveGeometricDets.end());
  
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
