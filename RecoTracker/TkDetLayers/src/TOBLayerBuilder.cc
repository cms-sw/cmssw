#include "RecoTracker/TkDetLayers/interface/TOBLayerBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkDetLayers/interface/TOBRodBuilder.h"

using namespace edm;
using namespace std;

TOBLayer* TOBLayerBuilder::build(const GeometricDet* aTOBLayer,
				 const TrackerGeometry* theGeomDetGeometry){

  vector<const GeometricDet*>  theGeometricDetRods = aTOBLayer->components();
  vector<const GeometricDet*> negativeZrods;
  vector<const GeometricDet*> positiveZrods;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricDetRods.begin();
      it!=theGeometricDetRods.end(); it++){
    if( (*it)->positionBounds().z() < 0) negativeZrods.push_back(*it);
    if( (*it)->positionBounds().z() > 0) positiveZrods.push_back(*it);
  }

  TOBRodBuilder myTOBRodBuilder;

  vector<const TOBRod*> theInnerRods;
  vector<const TOBRod*> theOuterRods;
  
  //LogDebug("TkDetLayers") << "positiveZrods[0]->positionBounds().perp(): " 
  //			  << positiveZrods[0]->positionBounds().perp() ;
  //LogDebug("TkDetLayers") << "positiveZrods[1]->positionBounds().perp(): " 
  //			  << positiveZrods[1]->positionBounds().perp() ;


  double meanR = (positiveZrods[0]->positionBounds().perp()+positiveZrods[1]->positionBounds().perp())/2;

  for(unsigned int index=0; index!=positiveZrods.size(); index++){
    if( positiveZrods[index]->positionBounds().phi() != negativeZrods[index]->positionBounds().phi()){
      edm::LogError("TkDetLayers") << "ERROR:rods don't have the same phi. exit!";
      break;      
    }
    
    if(positiveZrods[index]->positionBounds().perp() < meanR)
      theInnerRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						   positiveZrods[index],
						   theGeomDetGeometry)    );       
    if(positiveZrods[index]->positionBounds().perp() > meanR)
      theOuterRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						   positiveZrods[index],
						   theGeomDetGeometry)    );       
  }

  
  //LogDebug("TkDetLayers") << "theInnerRods.size(): " << theInnerRods.size() ;
  //LogDebug("TkDetLayers") << "theOuterRods.size(): " << theOuterRods.size() ;
  

  return new TOBLayer(theInnerRods,theOuterRods);
}
