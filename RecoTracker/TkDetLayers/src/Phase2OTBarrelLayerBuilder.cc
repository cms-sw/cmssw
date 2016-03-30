#include "Phase2OTBarrelLayerBuilder.h"
#include "Phase2OTtiltedBarrelLayer.h"
#include "Phase2OTBarrelRodBuilder.h"
#include "Phase2OTEndcapRingBuilder.h"

using namespace std;
using namespace edm;

Phase2OTBarrelLayer* Phase2OTBarrelLayerBuilder::build(const GeometricDet* aPhase2OTBarrelLayer,
						       const TrackerGeometry* theGeomDetGeometry)
{
  // This builder is very similar to TOBLayer one. Most of the code should be put in a 
  // common place.

  vector<const GeometricDet*>  theGeometricDets = aPhase2OTBarrelLayer->components();
  LogDebug("TkDetLayers") << "Phase2OTBarrelLayerBuilder with #Components: " << theGeometricDets.size() << std::endl;
  vector<const GeometricDet*>  theGeometricDetRods;
  vector<const GeometricDet*>  theGeometricDetRings;

  for(vector<const GeometricDet*>::const_iterator it = theGeometricDets.begin();
      it!=theGeometricDets.end(); it++){

    if( (*it)->type() == GeometricDet::ladder) {
      theGeometricDetRods.push_back(*it);
    } else if( (*it)->type() == GeometricDet::panel) {
      theGeometricDetRings.push_back(*it);
    } else {
      LogDebug("TkDetLayers") << "Phase2OTBarrelLayerBuilder with no Rods and no Rings! ";
    }
  }

  LogDebug("TkDetLayers") << "Phase2OTBarrelLayerBuilder with #Rods: " << theGeometricDetRods.size() << std::endl;

  Phase2OTBarrelRodBuilder myPhase2OTBarrelRodBuilder;

  vector<const Phase2OTBarrelRod*> theInnerRods;
  vector<const Phase2OTBarrelRod*> theOuterRods;

  // properly calculate the meanR value to separate rod in inner/outer.
  double meanR = 0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDetRods.begin();
      it!=theGeometricDetRods.end();it++){
    meanR = meanR + (*it)->positionBounds().perp();
  }
  meanR = meanR/theGeometricDetRods.size();

  for(unsigned int index=0; index!=theGeometricDetRods.size(); index++){    
    if(theGeometricDetRods[index]->positionBounds().perp() < meanR)
      theInnerRods.push_back(myPhase2OTBarrelRodBuilder.build(theGeometricDetRods[index],
								  theGeomDetGeometry)    );       

    if(theGeometricDetRods[index]->positionBounds().perp() > meanR)
      theOuterRods.push_back(myPhase2OTBarrelRodBuilder.build(theGeometricDetRods[index],
								  theGeomDetGeometry)    );       

  }

  if(theGeometricDetRings.empty()) return new Phase2OTBarrelLayer(theInnerRods,theOuterRods);
  
  LogDebug("TkDetLayers") << "Phase2OTBarrelLayerBuilder with #Rings: " << theGeometricDetRings.size() << std::endl;

  Phase2OTEndcapRingBuilder myPhase2OTEndcapRingBuilder;

  vector<const Phase2OTEndcapRing*> theNegativeRings;
  vector<const Phase2OTEndcapRing*> thePositiveRings;

  // properly calculate the meanR value to separate rod in inner/outer.
  double centralZ = 0.0;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricDetRings.begin();
      it!=theGeometricDetRings.end();it++){
    if((*it)->positionBounds().z() < centralZ)
      theNegativeRings.push_back(myPhase2OTEndcapRingBuilder.build( *it,theGeomDetGeometry));    
    if((*it)->positionBounds().z() > centralZ)
      thePositiveRings.push_back(myPhase2OTEndcapRingBuilder.build( *it,theGeomDetGeometry));    
  }

  return new Phase2OTtiltedBarrelLayer(theInnerRods,theOuterRods,theNegativeRings,thePositiveRings);

}

