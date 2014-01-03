#include "TECLayerBuilder.h"
#include "TECPetalBuilder.h"

using namespace edm;
using namespace std;

TECLayer* TECLayerBuilder::build(const GeometricDetPtr aTECLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  auto theGeometricDetPetals = aTECLayer->components();
  vector<const TECPetal*> theInnerPetals;
  vector<const TECPetal*> theOuterPetals;

  //edm::LogInfo(TkDetLayers) << "theGeometricDetPetals.size(): " << theGeometricDetPetals.size() ;
  
  double meanZ = ( theGeometricDetPetals.front()->positionBounds().z() + 
		   theGeometricDetPetals.back()->positionBounds().z() )/2;

  TECPetalBuilder myPetalBuilder;


  for(auto it=theGeometricDetPetals.cbegin();
      it!=theGeometricDetPetals.cend();it++){

    if( fabs((*it)->positionBounds().z()) < fabs(meanZ) ) 
      theInnerPetals.push_back(myPetalBuilder.build(*it,theGeomDetGeometry));

    if( fabs((*it)->positionBounds().z()) > fabs(meanZ) ) 
      theOuterPetals.push_back(myPetalBuilder.build(*it,theGeomDetGeometry));
  }
  
  return new TECLayer(theInnerPetals,theOuterPetals);
}
