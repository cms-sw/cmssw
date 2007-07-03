#include "RecoTracker/TkNavigation/interface/CosmicNavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/DiskLessInnerRadius.h"
#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "Utilities/General/interface/CMSexception.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

using namespace std;

CosmicNavigationSchool::CosmicNavigationSchool(const GeometricSearchTracker* theInputTracker,
					       const MagneticField* field)
{
  LogDebug("CosmicNavigationSchool") << "*********Running CosmicNavigationSchool***********" ;
  theBarrelLength = 0;theField = field; theTracker = theInputTracker;
  //byuild the fake layer to allow propagation from y > 0 to y < 0
  /*for (BNLCType::iterator i = theBarrelNLC.begin(); i != theBarrelNLC.end(); i++){
  	delete (*i);	
  }	
  theBarrelNLC.clear();
  for (FNLCType::iterator i = theForwardNLC.begin(); i != theForwardNLC.end(); i++){
        delete (*i);
  }
  theForwardNLC.clear();*/
  RectangularPlaneBounds bounds(700,700,1);
  GlobalPoint positionPlane(0,-4,0);
  TkRotation<float> rot(1,0,0,0,0,1,0,1,0); 
  //TkRotation<float> rot(1,0,0,0,1,0,0,0,1); 
  GlobalPoint positionCyl(0,0,0);
  //theFakeDetLayer = new FakeDetLayer(new BoundPlane(positionPlane, rot, &bounds), new BoundCylinder(positionCyl, rot, 2, SimpleCylinderBounds(2,2,-4,4)));
  theFakeDetLayer = new FakeDetLayer(new BoundPlane(positionPlane, rot, &bounds), new BoundCylinder(positionCyl, rot, 300, SimpleCylinderBounds(300,300,-500,500)));
  //theFakeDetLayer = new FakeDetLayer(&bp, &bc);
  std::cout << "theFakeDetLayer->specificSurface().radius() " << theFakeDetLayer->specificSurface().radius() << std::endl;
  // Get barrel layers
  vector<BarrelDetLayer*> blc = theTracker->barrelLayers();
  for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
    theBarrelLayers.push_back( (*i) );
  }
  theBarrelLayers.insert(theBarrelLayers.begin(),theFakeDetLayer);

  // get forward layers
  vector<ForwardDetLayer*> flc = theTracker->forwardLayers();
  for ( vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++) {
    theForwardLayers.push_back( (*i) );
  }

  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
                         not1(DetBelowZ(0)));
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
  theRightLayers = FDLC( middle, theForwardLayers.end());

  SymmetricLayerFinder symFinder( theForwardLayers);

  // only work on positive Z side; negative by mirror symmetry later
  linkBarrelLayers( symFinder);
  linkForwardLayers( symFinder);
  establishInverseRelations();
}
CosmicNavigationSchool::~CosmicNavigationSchool(){delete theFakeDetLayer;}

void CosmicNavigationSchool::
linkBarrelLayers( SymmetricLayerFinder& symFinder)
{
	//std::cout << "In link barrel layers" << std::endl; 
  // Link barrel layers outwards
  for ( BDLI i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    BDLC reachableBL;
    FDLC leftFL;
    FDLC rightFL;
    // link the fake layer to all barrel layers
    if (i == theBarrelLayers.begin()) {
	    linkToAllRegularBarrelLayer(reachableBL);
    } else {
	    // always add next barrel layer first
	    if ( i+1 != theBarrelLayers.end()) {reachableBL.push_back(*(i+1)); }
	    //if ( i+2 != theBarrelLayers.end()) {reachableBL.push_back(*(i+2));std::cout << "Adding " << (**(i+2)).specificSurface().radius() << " to " << (**i).specificSurface().radius() << std::endl;}
	 
	    // Add closest reachable forward layer (except for last BarrelLayer)
	    if (i != theBarrelLayers.end() - 1) {
	      linkNextForwardLayer( *i, rightFL);
	    }

	    // Add next BarrelLayer with length larger than the current BL
	    if ( i+2 < theBarrelLayers.end()) {
	      linkNextLargerLayer( i, theBarrelLayers.end(), reachableBL);
	    }
	    //reachableBL.push_back(theFakeDetLayer);	
    }
    //std::cout << "before constructing SimpleBarrelNavigableLayer" << std::endl;
    theBarrelNLC.push_back( new 
       SimpleBarrelNavigableLayer( *i, reachableBL,
				   symFinder.mirror(rightFL),
				   rightFL,theField, 5.));
    //std::cout << "after constructing SimpleBarrelNavigableLayer" << std::endl;
  }
  //std::cout << "done theBarrelNLC.push_back" << std::endl;  
}

void CosmicNavigationSchool::linkToAllRegularBarrelLayer( BDLC& reachableBL)
{

  for ( BDLI i = theBarrelLayers.begin()+1; i < theBarrelLayers.end(); i++) {
      //std::cout << "linking " << (*i)->specificSurface().radius() << " to " << (*theBarrelLayers.begin())->specificSurface().radius() << std::endl;
      reachableBL.push_back( *i);
  }
}

void CosmicNavigationSchool::linkNextBarrelLayer( ForwardDetLayer* fl,
                                                  BDLC& reachableBL)
{
  if ( fl->position().z() > barrelLength()) return;

  float outerRadius = fl->specificSurface().outerRadius();
  float zpos        = fl->position().z();
  for ( BDLI bli = theBarrelLayers.begin()+1; bli != theBarrelLayers.end(); bli++) {
    if ( outerRadius < (**bli).specificSurface().radius() &&
         zpos        < (**bli).surface().bounds().length() / 2.) {
      reachableBL.push_back( *bli);
      return;
    }
  }
}

void CosmicNavigationSchool::establishInverseRelations() {

  //std::cout << "In CosmicNavigationSchool::establishInverseRelations()" << std::endl;
  NavigationSetter setter(*this);

    // find for each layer which are the barrel and forward
    // layers that point to it
    typedef map<const DetLayer*, vector<BarrelDetLayer*>, less<const DetLayer*> > BarrelMapType;
    typedef map<const DetLayer*, vector<ForwardDetLayer*>, less<const DetLayer*> > ForwardMapType;


    BarrelMapType reachedBarrelLayersMap;
    ForwardMapType reachedForwardLayersMap;


    for ( BDLI bli = theBarrelLayers.begin();
        bli!=theBarrelLayers.end(); bli++) {
      DLC reachedLC = (**bli).nextLayers( insideOut);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        reachedBarrelLayersMap[*i].push_back( *bli);
      }
    }

    for ( FDLI fli = theForwardLayers.begin();
        fli!=theForwardLayers.end(); fli++) {
      DLC reachedLC = (**fli).nextLayers( insideOut);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        reachedForwardLayersMap[*i].push_back( *fli);
      }
    }


    vector<DetLayer*> lc = theTracker->allLayers();
    lc.insert(lc.begin(),theFakeDetLayer);
    for ( vector<DetLayer*>::iterator i = lc.begin(); i != lc.end(); i++) {
      SimpleNavigableLayer* navigableLayer =
	dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
	if (!navigableLayer) cout << " dynamic_cast<SimpleNavigableLayer*> failed: going to crash "  << std::endl; 	
          navigableLayer->setInwardLinks( reachedBarrelLayersMap[*i],reachedForwardLayersMap[*i] );
    }		
    for ( vector<DetLayer*>::iterator i = lc.begin(); i != lc.end(); i++) {	    
	//FakeDetLayer* layerToResize = dynamic_cast<FakeDetLayer*>(*i);
	//if (layerToResize) layerToResize->resizeCylinder(300);
	SimpleNavigableLayer* navigableLayer =
          dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
        //debug
	const BarrelDetLayer*  startLayerBarrel = dynamic_cast<const BarrelDetLayer*>(*i);
        const ForwardDetLayer* startLayerForward= dynamic_cast<const ForwardDetLayer*>(*i);
	if (startLayerBarrel) {
		edm::LogVerbatim("CosmicNavigationSchool") << "Starting from Barrel" << startLayerBarrel->specificSurface().radius() << " we can go to: " ;
	} else if (startLayerForward){
		edm::LogVerbatim("CosmicNavigationSchool") << "Starting from Forward" << startLayerForward->specificSurface().position().z() << " we can go to: " ;
	}
	edm::LogVerbatim("CosmicNavigationSchool") << "\tinsideOut: ";
	std::vector<const DetLayer*> inOutLayers = navigableLayer->nextLayers(insideOut);
	std::vector<const DetLayer*>::const_iterator il;
	for (il = inOutLayers.begin(); il != inOutLayers.end(); il++){
		const BarrelDetLayer*  layerBarrel = dynamic_cast<const BarrelDetLayer*>(*il);
      		const ForwardDetLayer* layerForward= dynamic_cast<const ForwardDetLayer*>(*il);
		if (layerBarrel){
			edm::LogVerbatim("CosmicNavigationSchool") << "Barrel "<<layerBarrel->specificSurface().radius() << ", ";
		} else if (layerForward){
			edm::LogVerbatim("CosmicNavigationSchool") << "Forward "<< layerForward->specificSurface().position().z() << ", ";
		}	
	}  
	//std::cout << std::endl;
	edm::LogVerbatim("CosmicNavigationSchool") << "\toutsideIn: ";
        std::vector<const DetLayer*> outInLayers = navigableLayer->nextLayers(outsideIn);
        for (il = outInLayers.begin(); il != outInLayers.end(); il++){
                const BarrelDetLayer*  layerBarrel = dynamic_cast<const BarrelDetLayer*>(*il);
                const ForwardDetLayer* layerForward= dynamic_cast<const ForwardDetLayer*>(*il);       
                if (layerBarrel){
                        edm::LogVerbatim("CosmicNavigationSchool") << "Barrel " <<  layerBarrel->specificSurface().radius() << ", ";
                } else if (layerForward){
                        edm::LogVerbatim("CosmicNavigationSchool") << "Forward " << layerForward->specificSurface().position().z() << ", ";
                }
        } 
        //std::cout << std::endl;
        //debug
    
    }
    
}

