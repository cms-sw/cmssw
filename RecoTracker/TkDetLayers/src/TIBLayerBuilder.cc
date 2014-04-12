#include "TIBLayerBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TIBRingBuilder.h"

using namespace edm;
using namespace std;

TIBLayer* TIBLayerBuilder::build(const GeometricDet* aTIBLayer,
				 const TrackerGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*> theGeometricRods = aTIBLayer->components();
  
  vector<vector<const GeometricDet*> > innerGeometricDetRings; 
  vector<vector<const GeometricDet*> > outerGeometricDetRings;
  
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
TIBLayerBuilder::constructRings(vector<const GeometricDet*>& theGeometricRods,
				vector<vector<const GeometricDet*> >& innerGeometricDetRings,
				vector<vector<const GeometricDet*> >& outerGeometricDetRings)
{
  double meanPerp=0;
  for(vector<const GeometricDet*>::const_iterator it=theGeometricRods.begin(); 
      it!= theGeometricRods.end();it++){
    meanPerp = meanPerp + (*it)->positionBounds().perp();
  }
  meanPerp = meanPerp/theGeometricRods.size();
  
  vector<const GeometricDet*> theInnerGeometricRods;
  vector<const GeometricDet*> theOuterGeometricRods;

  for(vector<const GeometricDet*>::const_iterator it=theGeometricRods.begin(); 
      it!= theGeometricRods.end();it++){
    if( (*it)->positionBounds().perp() < meanPerp) theInnerGeometricRods.push_back(*it);
    if( (*it)->positionBounds().perp() > meanPerp) theOuterGeometricRods.push_back(*it);
  }

  size_t innerLeftRodMaxSize  = 0;
  size_t innerRightRodMaxSize = 0;
  size_t outerLeftRodMaxSize  = 0;
  size_t outerRightRodMaxSize = 0;

  for(vector<const GeometricDet*>::const_iterator it=theInnerGeometricRods.begin(); 
      it!= theInnerGeometricRods.end();it++){
    if( (*it)->positionBounds().z() < 0) 
      innerLeftRodMaxSize  = max(innerLeftRodMaxSize,  (**it).components().size());
    if( (*it)->positionBounds().z() > 0) 
      innerRightRodMaxSize = max(innerRightRodMaxSize, (**it).components().size());
  }

  for(vector<const GeometricDet*>::const_iterator it=theOuterGeometricRods.begin(); 
      it!= theOuterGeometricRods.end();it++){
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
    innerGeometricDetRings.push_back(vector<const GeometricDet*>());
  }

  for(unsigned int i=0;i< (outerLeftRodMaxSize+outerRightRodMaxSize);i++){
    outerGeometricDetRings.push_back(vector<const GeometricDet*>());
  }
  
  for(unsigned int ringN = 0; ringN < innerLeftRodMaxSize; ringN++){
    for(vector<const GeometricDet*>::const_iterator it=theInnerGeometricRods.begin(); 
	it!= theInnerGeometricRods.end();it++){
      if( (*it)->positionBounds().z() < 0){
	if( (**it).components().size()>ringN)
	  innerGeometricDetRings[ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
  for(unsigned int ringN = 0; ringN < innerRightRodMaxSize; ringN++){
    for(vector<const GeometricDet*>::const_iterator it=theInnerGeometricRods.begin(); 
	it!= theInnerGeometricRods.end();it++){
      if( (*it)->positionBounds().z() > 0){
	if( (**it).components().size()>ringN)
	  innerGeometricDetRings[innerLeftRodMaxSize+ringN].push_back( (**it).components()[ringN] );
      }
    }
  }


  for(unsigned int ringN = 0; ringN < outerLeftRodMaxSize; ringN++){
    for(vector<const GeometricDet*>::const_iterator it=theOuterGeometricRods.begin(); 
	it!= theOuterGeometricRods.end();it++){
      if( (*it)->positionBounds().z() < 0){
	if( (**it).components().size()>ringN)
	  outerGeometricDetRings[ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
  for(unsigned int ringN = 0; ringN < outerRightRodMaxSize; ringN++){
    for(vector<const GeometricDet*>::const_iterator it=theOuterGeometricRods.begin(); 
	it!= theOuterGeometricRods.end();it++){
      if( (*it)->positionBounds().z() > 0){
	if( (**it).components().size()>ringN)
	  outerGeometricDetRings[outerLeftRodMaxSize+ringN].push_back( (**it).components()[ringN] );
      }
    }
  }
  
}



