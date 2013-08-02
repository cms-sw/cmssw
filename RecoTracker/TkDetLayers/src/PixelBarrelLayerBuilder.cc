#include "PixelBarrelLayerBuilder.h"
#include "PixelRodBuilder.h"

using namespace std;
using namespace edm;

PixelBarrelLayer* PixelBarrelLayerBuilder::build(const GeometricDet* aPixelBarrelLayer,
						 const TrackerGeometry* theGeomDetGeometry)
{
  // This builder is very similar to TOBLayer one. Most of the code should be put in a 
  // common place.

  vector<const GeometricDet*>  theGeometricDetRods = aPixelBarrelLayer->components();
  //edm::LogInfo(TkDetLayers) << "theGeometricDetRods has size: " << theGeometricDetRods.size() ;  
  

  PixelRodBuilder myPixelRodBuilder;

  vector<const PixelRod*> theInnerRods;
  vector<const PixelRod*> theOuterRods;

  // properly calculate the meanR value to separate rod in inner/outer.

  double meanR = 0;
  for (unsigned int index=0; index!=theGeometricDetRods.size(); index++)   meanR+=theGeometricDetRods[index]->positionBounds().perp();
  if (theGeometricDetRods.size()!=0)
    meanR/=(double) theGeometricDetRods.size();
  
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

