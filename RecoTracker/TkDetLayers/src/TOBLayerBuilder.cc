#include "RecoTracker/TkDetLayers/interface/TOBLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TOBRodBuilder.h"

TOBLayer* TOBLayerBuilder::build(const GeometricDet* aTOBLayer,
				 const TrackingGeometry* theGeomDetGeometry){

  vector<const GeometricDet*>  theGeometricDetRods = aTOBLayer->components();
  //cout << "theGeometricDetRods has size: " << theGeometricDetRods.size() << endl;  
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
  
  //cout << "positiveZrods[0]->positionBounds().perp(): " << positiveZrods[0]->positionBounds().perp() << endl;
  //cout << "positiveZrods[1]->positionBounds().perp(): " << positiveZrods[1]->positionBounds().perp() << endl;


  double meanR = (positiveZrods[0]->positionBounds().perp()+positiveZrods[1]->positionBounds().perp())/2;

  for(unsigned int index=0; index!=positiveZrods.size(); index++){
    if( positiveZrods[index]->positionBounds().phi() != negativeZrods[index]->positionBounds().phi()){
      cout << "ERROR:rods don't have the same phi. exit!" << endl;
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

  
  //cout << "theInnerRods.size(): " << theInnerRods.size() << endl;
  //cout << "theOuterRods.size(): " << theOuterRods.size() << endl;
  

  return new TOBLayer(theInnerRods,theOuterRods);
}
