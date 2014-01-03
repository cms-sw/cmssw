#include "TIBLayerBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TIBRingBuilder.h"

using namespace edm;
using namespace std;

TIBLayer* TIBLayerBuilder::build(const GeometricDetPtr aTIBLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  auto theGeometricRods = aTIBLayer->components();
  
  vector<vector<GeometricDetPtr> > innerGeometricDetRings; 
  vector<vector<GeometricDetPtr> > outerGeometricDetRings;
  
  constructRings(theGeometricRods,innerGeometricDetRings,outerGeometricDetRings);

  TIBRingBuilder myRingBuilder;

  vector<const TIBRing*> innerRings;
  vector<const TIBRing*> outerRings;

  for(unsigned int i=0; i<innerGeometricDetRings.size(); i++){
    innerRings.push_back(myRingBuilder.build(innerGeometricDetRings[i],theGeomDetGeometry));
    outerRings.push_back(myRingBuilder.build(outerGeometricDetRings[i],theGeomDetGeometry));
  }
    
  return new TIBLayer(innerRings,outerRings);
}




void 
TIBLayerBuilder::constructRings(vector<GeometricDetPtr>& theGeometricRods,
				vector<vector<GeometricDetPtr> >& innerGeometricDetRings,
				vector<vector<GeometricDetPtr> >& outerGeometricDetRings)
{
  double meanPerp=0;
  for(auto it=theGeometricRods.cbegin(); 
      it!= theGeometricRods.cend();it++){
    meanPerp = meanPerp + (*it)->positionBounds().perp();
  }
  meanPerp = meanPerp/theGeometricRods.size();
  
  vector<GeometricDetPtr> theInnerGeometricRods;
  vector<GeometricDetPtr> theOuterGeometricRods;

  for(auto it=theGeometricRods.cbegin(); 
      it!= theGeometricRods.cend();it++){
    if( (*it)->positionBounds().perp() < meanPerp) theInnerGeometricRods.push_back(*it);
    if( (*it)->positionBounds().perp() > meanPerp) theOuterGeometricRods.push_back(*it);
  }

  size_t innerLeftRodMaxSize  = 0;
  size_t innerRightRodMaxSize = 0;
  size_t outerLeftRodMaxSize  = 0;
  size_t outerRightRodMaxSize = 0;

  for(auto it=theInnerGeometricRods.cbegin(); 
      it!= theInnerGeometricRods.cend();it++){
    if( (*it)->positionBounds().z() < 0) 
      innerLeftRodMaxSize  = max(innerLeftRodMaxSize,  (**it).components().size());
    if( (*it)->positionBounds().z() > 0) 
      innerRightRodMaxSize = max(innerRightRodMaxSize, (**it).components().size());
  }

  for(auto it=theOuterGeometricRods.cbegin(); 
      it!= theOuterGeometricRods.cend();it++){
    if( (*it)->positionBounds().z() < 0) 
      outerLeftRodMaxSize  = max(outerLeftRodMaxSize,  (**it).components().size());
    if( (*it)->positionBounds().z() > 0) 
      outerRightRodMaxSize = max(outerRightRodMaxSize, (**it).components().size());
  }

  LogDebug("TkDetLayers") << "innerLeftRodMaxSize: " << innerLeftRodMaxSize ;
  LogDebug("TkDetLayers") << "innerRightRodMaxSize: " << innerRightRodMaxSize ;

  LogDebug("TkDetLayers") << "outerLeftRodMaxSize: " << outerLeftRodMaxSize ;
  LogDebug("TkDetLayers") << "outerRightRodMaxSize: " << outerRightRodMaxSize ;

  for(unsigned int i=0;i< (innerLeftRodMaxSize+innerRightRodMaxSize);i++){
    innerGeometricDetRings.push_back(vector<GeometricDetPtr>());
  }

  for(unsigned int i=0;i< (outerLeftRodMaxSize+outerRightRodMaxSize);i++){
    outerGeometricDetRings.push_back(vector<GeometricDetPtr>());
  }
  
  for(unsigned int ringN = 0; ringN < innerLeftRodMaxSize; ringN++){
    for(auto it=theInnerGeometricRods.cbegin(); 
	it!= theInnerGeometricRods.cend();it++){
      if( (*it)->positionBounds().z() < 0){
	if( (**it).components().size()>ringN)
	  innerGeometricDetRings[ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
  for(unsigned int ringN = 0; ringN < innerRightRodMaxSize; ringN++){
    for(auto it=theInnerGeometricRods.cbegin(); 
	it!= theInnerGeometricRods.cend();it++){
      if( (*it)->positionBounds().z() > 0){
	if( (**it).components().size()>ringN)
	  innerGeometricDetRings[innerLeftRodMaxSize+ringN].push_back( (**it).components()[ringN] );
      }
    }
  }


  for(unsigned int ringN = 0; ringN < outerLeftRodMaxSize; ringN++){
    for(auto it=theOuterGeometricRods.cbegin(); 
	it!= theOuterGeometricRods.cend();it++){
      if( (*it)->positionBounds().z() < 0){
	if( (**it).components().size()>ringN)
	  outerGeometricDetRings[ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
  for(unsigned int ringN = 0; ringN < outerRightRodMaxSize; ringN++){
    for(auto it=theOuterGeometricRods.cbegin(); 
	it!= theOuterGeometricRods.cend();it++){
      if( (*it)->positionBounds().z() > 0){
	if( (**it).components().size()>ringN)
	  outerGeometricDetRings[outerLeftRodMaxSize+ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
}



