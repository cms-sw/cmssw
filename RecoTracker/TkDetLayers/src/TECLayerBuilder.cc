#include "TECLayerBuilder.h"
#include "TECPetalBuilder.h"

using namespace edm;
using namespace std;

TECLayer* TECLayerBuilder::build(const GeometricDet* aTECLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*>  theGeometricDetPetals = aTECLayer->components();
  vector<const TECPetal*> theInnerPetals;
  vector<const TECPetal*> theOuterPetals;

  //edm::LogInfo(TkDetLayers) << "theGeometricDetPetals.size(): " << theGeometricDetPetals.size() ;
  
  double meanZ = ( theGeometricDetPetals.front()->positionBounds().z() + 
		   theGeometricDetPetals.back()->positionBounds().z() )/2;

  TECPetalBuilder myPetalBuilder;


  for(vector<const GeometricDet*>::const_iterator it=theGeometricDetPetals.begin();
      it!=theGeometricDetPetals.end();it++){

    if( fabs((*it)->positionBounds().z()) < fabs(meanZ) ) 
      theInnerPetals.push_back(myPetalBuilder.build(*it,theGeomDetGeometry));

    if( fabs((*it)->positionBounds().z()) > fabs(meanZ) ) 
      theOuterPetals.push_back(myPetalBuilder.build(*it,theGeomDetGeometry));
  }
  
  return new TECLayer(theInnerPetals,theOuterPetals);
}
