#include "Phase2OTBarrelLayerBuilder.h"
#include "Phase2OTBarrelRodBuilder.h"

using namespace std;
using namespace edm;

Phase2OTBarrelLayer* Phase2OTBarrelLayerBuilder::build(const GeometricDet* aPhase2OTBarrelLayer,
						 const TrackerGeometry* theGeomDetGeometry)
{
  // This builder is very similar to TOBLayer one. Most of the code should be put in a 
  // common place.

  vector<const GeometricDet*>  theGeometricDetRods = aPhase2OTBarrelLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricDetRods has size: " << theGeometricDetRods.size() ;  
  

  Phase2OTBarrelRodBuilder myPhase2OTBarrelRodBuilder;

  vector<const Phase2OTBarrelRod*> theInnerRods;
  vector<const Phase2OTBarrelRod*> theOuterRods;

  // properly calculate the meanR value to separate rod in inner/outer.

  double meanR = 0;
  for (unsigned int index=0; index!=theGeometricDetRods.size(); index++)   meanR+=theGeometricDetRods[index]->positionBounds().perp();
  if (theGeometricDetRods.size()!=0)
    meanR/=(double) theGeometricDetRods.size();
  
  for(unsigned int index=0; index!=theGeometricDetRods.size(); index++){    
    if(theGeometricDetRods[index]->positionBounds().perp() < meanR)
      theInnerRods.push_back(myPhase2OTBarrelRodBuilder.build(theGeometricDetRods[index],
								  theGeomDetGeometry)    );       

    if(theGeometricDetRods[index]->positionBounds().perp() > meanR)
      theOuterRods.push_back(myPhase2OTBarrelRodBuilder.build(theGeometricDetRods[index],
								  theGeomDetGeometry)    );       

  }
  
  return new Phase2OTBarrelLayer(theInnerRods,theOuterRods);

}

