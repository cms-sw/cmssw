#include "TOBLayerBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TOBRodBuilder.h"

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
    if( (*it)->positionBounds().z() >= 0) positiveZrods.push_back(*it);
  }

  TOBRodBuilder myTOBRodBuilder;

  vector<const TOBRod*> theInnerRods;
  vector<const TOBRod*> theOuterRods;
  
  //LogDebug("TkDetLayers") << "positiveZrods[0]->positionBounds().perp(): " 
  //			  << positiveZrods[0]->positionBounds().perp() ;
  //LogDebug("TkDetLayers") << "positiveZrods[1]->positionBounds().perp(): " 
  //			  << positiveZrods[1]->positionBounds().perp() ;

  double positiveMeanR = 0;
  if(!positiveZrods.empty()){
    for(unsigned int index=0; index!=positiveZrods.size(); index++){
      positiveMeanR += positiveZrods[index]->positionBounds().perp();
    }
    positiveMeanR = positiveMeanR/positiveZrods.size();
  }


  double negativeMeanR = 0;
  if(!negativeZrods.empty()){
    for(unsigned int index=0; index!=negativeZrods.size(); index++){
      negativeMeanR += negativeZrods[index]->positionBounds().perp();
    }
    negativeMeanR = negativeMeanR/negativeZrods.size();
  }

  if(!positiveZrods.empty() && !negativeZrods.empty()){
    for(unsigned int index=0; index!=positiveZrods.size(); index++){
      if( positiveZrods[index]->positionBounds().phi() != negativeZrods[index]->positionBounds().phi()){
	edm::LogError("TkDetLayers") << "ERROR:rods don't have the same phi. exit!";
	break;      
      }

      if(positiveZrods[index]->positionBounds().perp() < positiveMeanR)
	theInnerRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						     positiveZrods[index],
						     theGeomDetGeometry)    );       
      if(positiveZrods[index]->positionBounds().perp() >= positiveMeanR)
	theOuterRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						     positiveZrods[index],
						     theGeomDetGeometry)    );
    }
  } else{
    if(!positiveZrods.empty()){
      for(unsigned int index=0; index!=positiveZrods.size(); index++){
	if(positiveZrods[index]->positionBounds().perp() < positiveMeanR)
	  theInnerRods.push_back(myTOBRodBuilder.build(nullptr,
						       positiveZrods[index],
						       theGeomDetGeometry)    );       
	if(positiveZrods[index]->positionBounds().perp() >= positiveMeanR)
	  theOuterRods.push_back(myTOBRodBuilder.build(nullptr,
						       positiveZrods[index],
						       theGeomDetGeometry)    );       
      }
    }
    if(!negativeZrods.empty()){
      for(unsigned int index=0; index!=negativeZrods.size(); index++){
	if(negativeZrods[index]->positionBounds().perp() < negativeMeanR)
	  theInnerRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						       nullptr,
						       theGeomDetGeometry)    );       
	if(negativeZrods[index]->positionBounds().perp() >= negativeMeanR)
	  theOuterRods.push_back(myTOBRodBuilder.build(negativeZrods[index],
						       nullptr,
						       theGeomDetGeometry)    );
      }
    }
  }
  
  //LogDebug("TkDetLayers") << "theInnerRods.size(): " << theInnerRods.size() ;
  //LogDebug("TkDetLayers") << "theOuterRods.size(): " << theOuterRods.size() ;
  

  return new TOBLayer(theInnerRods,theOuterRods);
}
