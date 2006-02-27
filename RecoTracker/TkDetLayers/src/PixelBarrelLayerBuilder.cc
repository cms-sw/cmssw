#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/PixelRodBuilder.h"

PixelBarrelLayer* PixelBarrelLayerBuilder::build(const GeometricDet* aPixelBarrelLayer,
						 const TrackingGeometry* theGeomDetGeometry)
{
  // This builder is very similar to TOBLayer one. Most of the code should be put in a 
  // common place.

  vector<const GeometricDet*>  theGeometricDetRods = aPixelBarrelLayer->components();
  //cout << "theGeometricDetRods has size: " << theGeometricDetRods.size() << endl;  
  

  PixelRodBuilder myPixelRodBuilder;

  vector<const PixelRod*> theInnerRods;
  vector<const PixelRod*> theOuterRods;

  double meanR = (theGeometricDetRods[0]->positionBounds().perp()+theGeometricDetRods[1]->positionBounds().perp())/2;
  
  for(unsigned int index=0; index!=theGeometricDetRods.size(); index++){    
    if(theGeometricDetRods[index]->positionBounds().perp() < meanR)
      theInnerRods.push_back(myPixelRodBuilder.build(theGeometricDetRods[index],
						     theGeomDetGeometry)    );       

    if(theGeometricDetRods[index]->positionBounds().perp() > meanR)
      theOuterRods.push_back(myPixelRodBuilder.build(theGeometricDetRods[index],
						     theGeomDetGeometry)    );       

  }
  
  return new PixelBarrelLayer(theInnerRods,theOuterRods);

}

