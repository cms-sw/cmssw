#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

#include "PixelBarrelLayerBuilder.h"
#include "PixelForwardLayerBuilder.h"
#include "TIBLayerBuilder.h"
#include "TOBLayerBuilder.h"
#include "TIDLayerBuilder.h"
#include "TECLayerBuilder.h"

#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "DataFormats/Common/interface/Trie.h"
#include <boost/function.hpp>
#include <boost/bind.hpp>

using namespace std;

GeometricSearchTracker*
GeometricSearchTrackerBuilder::build(const GeometricDet* theGeometricTracker,
				     const TrackerGeometry* theGeomDetGeometry)
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
  
  using namespace trackerTrie;

  //-- future code
  // create a Trie
  DetTrie trie(0);

  // to be moved elsewhere
  {
    const TrackingGeometry::DetUnitContainer&  modules = theGeomDetGeometry->detUnits(); 
    typedef TrackingGeometry::DetUnitContainer::const_iterator Iter;
    Iter b=modules.begin();
    Iter e=modules.end();
    Iter last;
    try {
      for(;b!=e; ++b) {
	last = b;
	unsigned int rawid = (*b)->geographicalId().rawId();
	trie.insert(trackerHierarchy(rawid), *b); 
      }
    }
    catch(edm::Exception const & e) {
      std::cout << "in filling " << e.what() << std::endl;
      unsigned int rawid = (*last)->geographicalId().rawId();
      int subdetid = (*last)->geographicalId().subdetId();
      std::cout << rawid << " " << subdetid << std::endl;
    }  
  }

  // layers "ids"
  unsigned int layerId[] = {1,3,5,21,22,41,42,61,62};
  //  boost::function<void(trackerTrie::Node const &)> fun[9]; 
  /*
	thePxlBarLayers.push_back( aPixelBarrelLayerBuilder.build(*p) );
	theTIBLayers.push_back( aTIBLayerBuilder.build(*p) );
	theTOBLayers.push_back( aTOBLayerBuilder.build(*p) );
	theNegPxlFwdLayers.push_back( aPixelForwardLayerBuilder.build(*p) );
	thePosPxlFwdLayers.push_back( aPixelForwardLayerBuilder.build(*p) );
	theNegTIDLayers.push_back( aTIDLayerBuilder.build(*p) );
	thePosTIDLayers.push_back( aTIDLayerBuilder.build(*p) );
	theNegTECLayers.push_back( aTECLayerBuilder.build(*p) );
	thePosTECLayers.push_back( aTECLayerBuilder.build(*p) );
  */
   

  for (int i=0;i<9;i++) {
    std::string s;
    if (layerId[i]>9) s+=char(layerId[i]/10); 
    s+=char(layerId[i]%10);
    node_iterator e;	
    node_iterator p(trie.node(s));
    for (;p!=e;++p) {
      //    fun[i](*p);
    }
  }    


  // current code
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

    
    if( (*it)->type() == GeometricDet::PixelEndCap || (*it)->type() == GeometricDet::PixelEndCapPhase1 ){
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
  

  return new GeometricSearchTracker(thePxlBarLayers,theTIBLayers,theTOBLayers,
				    theNegPxlFwdLayers,theNegTIDLayers,theNegTECLayers,
				    thePosPxlFwdLayers,thePosTIDLayers,thePosTECLayers);
}
