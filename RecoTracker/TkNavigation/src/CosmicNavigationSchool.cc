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

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

using namespace std;

CosmicNavigationSchool::CosmicNavigationSchoolConfiguration::CosmicNavigationSchoolConfiguration(const edm::ParameterSet conf){
  noPXB=conf.getParameter<bool>("noPXB");
  noPXF=conf.getParameter<bool>("noPXF");
  noTIB=conf.getParameter<bool>("noTIB");
  noTID=conf.getParameter<bool>("noTID");
  noTOB=conf.getParameter<bool>("noTOB");
  noTEC=conf.getParameter<bool>("noTEC");
  self = conf.getParameter<bool>("selfSearch");
  allSelf = conf.getParameter<bool>("allSelf");
}

CosmicNavigationSchool::CosmicNavigationSchool(const GeometricSearchTracker* theInputTracker,
					       const MagneticField* field)
{
  build(theInputTracker, field, CosmicNavigationSchoolConfiguration());
}
 
void CosmicNavigationSchool::build(const GeometricSearchTracker* theInputTracker,
				   const MagneticField* field,
				   const CosmicNavigationSchoolConfiguration conf)
{
  LogTrace("CosmicNavigationSchool") << "*********Running CosmicNavigationSchool***********" ;	
  theBarrelLength = 0;theField = field; theTracker = theInputTracker;

  theAllDetLayersInSystem=&theInputTracker->allLayers();

  // Get barrel layers
  vector<BarrelDetLayer*> blc = theTracker->barrelLayers();
  for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
    if (conf.noPXB && (*i)->subDetector() == GeomDetEnumerators::PixelBarrel) continue;
    if (conf.noTOB && (*i)->subDetector() == GeomDetEnumerators::TOB) continue;
    if (conf.noTIB && (*i)->subDetector() == GeomDetEnumerators::TIB) continue;
    theBarrelLayers.push_back( (*i) );
  }

  // get forward layers
  vector<ForwardDetLayer*> flc = theTracker->forwardLayers();
  for ( vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++) {
    if (conf.noPXF && (*i)->subDetector() == GeomDetEnumerators::PixelEndcap) continue;
    if (conf.noTEC && (*i)->subDetector() == GeomDetEnumerators::TEC) continue;
    if (conf.noTID && (*i)->subDetector() == GeomDetEnumerators::TID) continue;
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
  establishInverseRelations( symFinder );

  if (conf.self){

    // set the self search by hand
    NavigationSetter setter(*this);

    //add TOB1->TOB1 inward link
    const std::vector< BarrelDetLayer * > &  tobL = theInputTracker->tobLayers();
    if (tobL.size()>=1){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all TOB self search.";
	for (std::vector< BarrelDetLayer * >::const_iterator lIt = tobL.begin(); lIt!=tobL.end(); ++lIt)
	  dynamic_cast<SimpleNavigableLayer*>((*lIt)->navigableLayer())->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer = dynamic_cast<SimpleNavigableLayer*>(tobL.front()->navigableLayer());
	LogDebug("CosmicNavigationSchool")<<" adding TOB1 to TOB1.";
	navigableLayer->theSelfSearch = true;
      }
    }
    const std::vector< BarrelDetLayer * > &  tibL = theInputTracker->tibLayers();
    if (tibL.size()>=1){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all TIB self search.";
	for (std::vector< BarrelDetLayer * >::const_iterator lIt = tibL.begin(); lIt!=tibL.end(); ++lIt)
	  dynamic_cast<SimpleNavigableLayer*>((*lIt)->navigableLayer())->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer = dynamic_cast<SimpleNavigableLayer*>(tibL.front()->navigableLayer());
	LogDebug("CosmicNavigationSchool")<<" adding tib1 to tib1.";
	navigableLayer->theSelfSearch = true;
      }
    }
    const std::vector< BarrelDetLayer * > &  pxbL = theInputTracker->pixelBarrelLayers();
    if (pxbL.size()>=1){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all PXB self search.";
        for (std::vector< BarrelDetLayer * >::const_iterator lIt = pxbL.begin(); lIt!=pxbL.end(); ++lIt)
          dynamic_cast<SimpleNavigableLayer*>((*lIt)->navigableLayer())->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer = dynamic_cast<SimpleNavigableLayer*>(pxbL.front()->navigableLayer());
	LogDebug("CosmicNavigationSchool")<<" adding pxb1 to pxb1.";
	navigableLayer->theSelfSearch = true;
      }
    }
  }
}

void CosmicNavigationSchool::
linkBarrelLayers( SymmetricLayerFinder& symFinder)
{
  //identical to the SimpleNavigationSchool one, but it allows crossing over the tracker
  //is some non-standard link is needed, it should probably be added here
  
  // Link barrel layers outwards
  for ( BDLI i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    BDLC reachableBL;
    FDLC leftFL;
    FDLC rightFL;

    // always add next barrel layer first
    if ( i+1 != theBarrelLayers.end()) reachableBL.push_back(*(i+1));

    // Add closest reachable forward layer (except for last BarrelLayer)
    if (i != theBarrelLayers.end() - 1) {
      linkNextForwardLayer( *i, rightFL);
    }

    // Add next BarrelLayer with length larger than the current BL
    if ( i+2 < theBarrelLayers.end()) {
      linkNextLargerLayer( i, theBarrelLayers.end(), reachableBL);
    }

    theBarrelNLC.push_back( new
       SimpleBarrelNavigableLayer( *i, reachableBL,
                                   symFinder.mirror(rightFL),
                                   rightFL,theField, 5.,false));
  }
}


void CosmicNavigationSchool::establishInverseRelations(SymmetricLayerFinder& symFinder) {
    
    //again: standard part is identical to SimpleNavigationSchool one. 
    //After the standard link, special outsideIn links are added  

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
    for ( vector<DetLayer*>::iterator i = lc.begin(); i != lc.end(); i++) {
      SimpleNavigableLayer* navigableLayer = dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
      if (navigableLayer)
	navigableLayer->setInwardLinks( reachedBarrelLayersMap[*i],reachedForwardLayersMap[*i] );
    }	
    //buildAdditionalBarrelLinks();
    buildAdditionalForwardLinks(symFinder); 

}


void CosmicNavigationSchool::buildAdditionalBarrelLinks(){
    for ( vector<BarrelDetLayer*>::iterator i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
      SimpleNavigableLayer* navigableLayer =
        dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
        if (i+1 != theBarrelLayers.end() )navigableLayer->setAdditionalLink(*(i+1), outsideIn);
    }
}


void CosmicNavigationSchool::buildAdditionalForwardLinks(SymmetricLayerFinder& symFinder){
    //the first layer of FPIX should not check the crossing side (since there are no inner layers to be tryed first)
    SimpleNavigableLayer* firstR = dynamic_cast<SimpleNavigableLayer*>(theRightLayers.front()->navigableLayer());
    SimpleNavigableLayer* firstL = dynamic_cast<SimpleNavigableLayer*>(theLeftLayers.front()->navigableLayer());
    firstR->setCheckCrossingSide(false);	
    firstL->setCheckCrossingSide(false);	
    	
    for ( vector<ForwardDetLayer*>::iterator i = theRightLayers.begin(); i != theRightLayers.end(); i++){
	//look for first bigger barrel layer and link to it outsideIn
	SimpleForwardNavigableLayer*  nfl = dynamic_cast<SimpleForwardNavigableLayer*>((*i)->navigableLayer());
        SimpleForwardNavigableLayer* mnfl = dynamic_cast<SimpleForwardNavigableLayer*>(symFinder.mirror(*i)->navigableLayer());
	for (vector<BarrelDetLayer*>::iterator j = theBarrelLayers.begin(); j != theBarrelLayers.end(); j++){
	    if ((*i)->specificSurface().outerRadius() < (*j)->specificSurface().radius() && 
	         fabs((*i)->specificSurface().position().z()) < (*j)->surface().bounds().length()/2.){ 
	        nfl ->setAdditionalLink(*j, outsideIn);
		mnfl->setAdditionalLink(*j, outsideIn);	
		break;
	    }	
	}
    }	  	
}

