#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/PixelForwardLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TOBLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TIDLayerBuilder.h"
#include "RecoTracker/TkDetLayers/interface/TECLayerBuilder.h"

GeometricSearchTracker*
GeometricSearchTrackerBuilder::build(const GeometricDet* theGeometricTracker,
				     const TrackingGeometry* theGeomDetGeometry)
{
  PixelBarrelLayerBuilder aPixelBarrelLayerBuilder;
  PixelForwardLayerBuilder aPixelForwardLayerBuilder;
  TIBLayerBuilder aTIBLayerBuilder;
  TOBLayerBuilder aTOBLayerBuilder;
  TIDLayerBuilder aTIDLayerBuilder;
  TECLayerBuilder aTECLayerBuilder;

  vector<BarrelDetLayer*>  thePxlBarLayers;
  vector<BarrelDetLayer*>  theTIBLayers;
  vector<BarrelDetLayer*>  theTOBLayers;
  vector<ForwardDetLayer*> theNegPxlFwdLayers;
  vector<ForwardDetLayer*> thePosPxlFwdLayers;
  vector<ForwardDetLayer*> theNegTIDLayers;
  vector<ForwardDetLayer*> thePosTIDLayers;
  vector<ForwardDetLayer*> theNegTECLayers;
  vector<ForwardDetLayer*> thePosTECLayers;
  

  vector<const GeometricDet*> theGeometricDetLayers = theGeometricTracker->components();
  for(vector<const GeometricDet*>::const_iterator it=theGeometricDetLayers.begin();
      it!=theGeometricDetLayers.end(); it++){

    if( (*it)->type() == GeometricDet::PixelBarrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=thePxlBarGeometricDetLayers.begin();
	  it2!=thePxlBarGeometricDetLayers.end(); it2++){
	thePxlBarLayers.push_back( aPixelBarrelLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }
    
    if( (*it)->type() == GeometricDet::TIB) {
      vector<const GeometricDet*> theTIBGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=theTIBGeometricDetLayers.begin();
	  it2!=theTIBGeometricDetLayers.end(); it2++){
	theTIBLayers.push_back( aTIBLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }

    if( (*it)->type() == GeometricDet::TOB) {
      vector<const GeometricDet*> theTOBGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=theTOBGeometricDetLayers.begin();	  
      it2!=theTOBGeometricDetLayers.end(); it2++){
	theTOBLayers.push_back( aTOBLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }

    
    if( (*it)->type() == GeometricDet::PixelEndCap){
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=thePxlFwdGeometricDetLayers.begin();
	  it2!=thePxlFwdGeometricDetLayers.end(); it2++){
	if((*it2)->positionBounds().z() < 0)
	  theNegPxlFwdLayers.push_back( aPixelForwardLayerBuilder.build(*it2,theGeomDetGeometry) );
	if((*it2)->positionBounds().z() > 0)
	  thePosPxlFwdLayers.push_back( aPixelForwardLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }
    


    if( (*it)->type() == GeometricDet::TID){
      vector<const GeometricDet*> theTIDGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=theTIDGeometricDetLayers.begin();
      it2!=theTIDGeometricDetLayers.end(); it2++){
	if((*it2)->positionBounds().z() < 0)
	  theNegTIDLayers.push_back( aTIDLayerBuilder.build(*it2,theGeomDetGeometry) );
	if((*it2)->positionBounds().z() > 0)
	  thePosTIDLayers.push_back( aTIDLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }

    if( (*it)->type() == GeometricDet::TEC) {
      vector<const GeometricDet*> theTECGeometricDetLayers = (*it)->components();
      for(vector<const GeometricDet*>::const_iterator it2=theTECGeometricDetLayers.begin();
      it2!=theTECGeometricDetLayers.end(); it2++){
	if((*it2)->positionBounds().z() < 0)
	  theNegTECLayers.push_back( aTECLayerBuilder.build(*it2,theGeomDetGeometry) );
	if((*it2)->positionBounds().z() > 0)
	  thePosTECLayers.push_back( aTECLayerBuilder.build(*it2,theGeomDetGeometry) );
      }
    }


  }


  
  vector<BarrelDetLayer*>  bl(thePxlBarLayers.begin(),thePxlBarLayers.end());
  bl.insert(bl.end(),theTIBLayers.begin(),theTIBLayers.end());
  bl.insert(bl.end(),theTOBLayers.begin(),theTOBLayers.end());
  vector<ForwardDetLayer*> negFl(theNegPxlFwdLayers.begin(),theNegPxlFwdLayers.end());
  negFl.insert(negFl.end(),theNegTIDLayers.begin(),theNegTIDLayers.end());
  negFl.insert(negFl.end(),theNegTECLayers.begin(),theNegTECLayers.end());
  vector<ForwardDetLayer*> posFl(thePosPxlFwdLayers.begin(),thePosPxlFwdLayers.end());
  posFl.insert(posFl.end(),thePosTIDLayers.begin(),thePosTIDLayers.end());
  posFl.insert(posFl.end(),thePosTECLayers.begin(),thePosTECLayers.end());


  cout << "n pxlBarLayers: " << thePxlBarLayers.size() << endl;
  cout << "n tibLayers:    " << theTIBLayers.size() << endl;
  cout << "n tobLayers:    " << theTOBLayers.size() << endl;
  cout << "n negPxlFwdLayers: " << theNegPxlFwdLayers.size() << endl;
  cout << "n negPxlFwdLayers: " << thePosPxlFwdLayers.size() << endl;
  cout << "n negTidLayers: " << theNegTIDLayers.size() << endl;
  cout << "n posTidLayers: " << thePosTIDLayers.size() << endl;
  cout << "n negTecLayers: " << theNegTECLayers.size() << endl;
  cout << "n posTecLayers: " << thePosTECLayers.size() << endl;

  cout << "n barreLayers:  " << bl.size() << endl;
  cout << "n negforwardLayers: " << negFl.size() << endl;
  cout << "n posForwardLayers: " << posFl.size() << endl;

  return new GeometricSearchTracker(bl,negFl,posFl);
}
