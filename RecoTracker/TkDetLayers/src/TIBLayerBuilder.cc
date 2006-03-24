#include "RecoTracker/TkDetLayers/interface/TIBLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TIBRingBuilder.h"

TIBLayer* TIBLayerBuilder::build(const GeometricDet* aTIBLayer,
				 const TrackingGeometry* theGeomDetGeometry)
{
  vector<const GeometricDet*> theGeometricRods = aTIBLayer->components();
  
  vector<vector<const GeometricDet*> > innerGeometricDetRings; 
  vector<vector<const GeometricDet*> > outerGeometricDetRings;

  for(int i=0;i<6;i++){
    innerGeometricDetRings.push_back(vector<const GeometricDet*>());
    outerGeometricDetRings.push_back(vector<const GeometricDet*>());
  }
  
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
  

  for(vector<const GeometricDet*>::const_iterator it=theInnerGeometricRods.begin(); 
      it!= theInnerGeometricRods.end();it++){
    if( (*it)->positionBounds().z() < 0) {
      innerGeometricDetRings[0].push_back( (**it).components()[0] );
      innerGeometricDetRings[1].push_back( (**it).components()[1] );
      innerGeometricDetRings[2].push_back( (**it).components()[2] );
    }
    if( (*it)->positionBounds().z() > 0) {
      innerGeometricDetRings[3].push_back( (**it).components()[0] );
      innerGeometricDetRings[4].push_back( (**it).components()[1] );
      innerGeometricDetRings[5].push_back( (**it).components()[2] );
    }    
  }
  

  for(vector<const GeometricDet*>::const_iterator it=theOuterGeometricRods.begin(); 
      it!= theOuterGeometricRods.end();it++){
    if( (*it)->positionBounds().z() < 0) {
      outerGeometricDetRings[0].push_back( (**it).components()[0] );
      outerGeometricDetRings[1].push_back( (**it).components()[1] );
      outerGeometricDetRings[2].push_back( (**it).components()[2] );
    }
    if( (*it)->positionBounds().z() > 0) {
      outerGeometricDetRings[3].push_back( (**it).components()[0] );
      outerGeometricDetRings[4].push_back( (**it).components()[1] );
      outerGeometricDetRings[5].push_back( (**it).components()[2] );
    }    
  }

}
