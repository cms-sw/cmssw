#include "RecoTracker/TkNavigation/interface/CfgNavigationSchool.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CfgNavigationSchool::CfgNavigationSchool(const edm::ParameterSet & cfg,
					 const GeometricSearchTracker* theInputTracker,
					 const MagneticField* field){
  //some genericinitialisation
  theTracker=theInputTracker;
  theField=field;
  theAllDetLayersInSystem=&theInputTracker->allLayers();
  // Get barrel layers
  std::vector<BarrelDetLayer*> blc = theTracker->barrelLayers(); 
  for ( std::vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++)
    theBarrelLayers.push_back( (*i) );
  // get forward layers
  std::vector<ForwardDetLayer*> flc = theTracker->forwardLayers(); 
  for ( std::vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++)
    theForwardLayers.push_back( (*i) );

  std::vector< std::string > names;
  cfg.getParameterSetNames(names);

  bool inverseRelationShip = true;
  for (unsigned int iN=0;iN!=names.size();++iN){
    edm::ParameterSet pset=cfg.getParameter<edm::ParameterSet>(names[iN]);
    std::vector<std::string> OUT = pset.getParameter<std::vector<std::string> >("OUT");
    //will not do automatic inverse relation is any IN is specified
    if ( pset.exists("IN") ) inverseRelationShip = false;

    BDLC reachableBL;
    FDLC reachableFL;

    //create the OUT links
    for (unsigned int iOut=0;iOut!=OUT.size();++iOut)
      addLayer(OUT[iOut], reachableBL, reachableFL);

    makeFwdLinks(names[iN],reachableBL,reachableFL);
  }

  //set the navigation to be able to access the NavigableLayer from the DetLayer itself
  NavigationSetter setter(*this); 

  if( inverseRelationShip ){
    establishInverseRelations();
  }else{
    //set it by hand in the configuration
    for (unsigned int iN=0;iN!=names.size();++iN){
      edm::ParameterSet pset=cfg.getParameter<edm::ParameterSet>(names[iN]);
      std::vector<std::string> IN = pset.getParameter<std::vector<std::string> >("IN");
      
      BDLC reachableBL;
      FDLC reachableFL;
      
      //create the IN links
      for (unsigned int iIn=0;iIn!=IN.size();++iIn)
	addLayer(IN[iIn], reachableBL, reachableFL);
      
      makeBwdLinks(names[iN],reachableBL,reachableFL);
    }
  }
}

void CfgNavigationSchool::makeFwdLinks(std::string & lname, BDLC & reachableBL, FDLC & reachableFL){
  DetLayer * l = layer(lname);
  if (l->location() == GeomDetEnumerators::barrel){
    //split the FL into left and right.

    FDLI middle = find_if( reachableFL.begin(), reachableFL.end(),
			   not1(DetBelowZ(0)));
    FDLC leftFL(reachableFL.begin(), middle);
    FDLC rightFL(middle, reachableFL.end());    

    BarrelDetLayer * bl = dynamic_cast<BarrelDetLayer *>(l);
    theBarrelNLC.push_back( new SimpleBarrelNavigableLayer(bl,
							   reachableBL,
							   rightFL,leftFL,
							   theField,
							   5.));
  }
  else{
    ForwardDetLayer * fwdL = dynamic_cast<ForwardDetLayer *>(l);
    theForwardNLC.push_back( new SimpleForwardNavigableLayer(fwdL,
							     reachableBL,
							     reachableFL,
							     theField,
							     5.));
  }
}

void CfgNavigationSchool::makeBwdLinks(std::string & lname, BDLC & reachableBL, FDLC & reachableFL){
  DetLayer * l = layer(lname);
  SimpleNavigableLayer * nl = dynamic_cast<SimpleNavigableLayer*>(l->navigableLayer());
  if (nl) nl->setInwardLinks(reachableBL,reachableFL);
  else 
    edm::LogError("CfgNavigationSchool")<<"a layer is not casting to SimpleNavigableLayer.";
}


void CfgNavigationSchool::addLayer(std::string & lname, BDLC & reachableBL, FDLC & reachableFL){
  DetLayer * l = layer(lname);
  if (l->location() == GeomDetEnumerators::barrel)
    reachableBL.push_back(dynamic_cast<BarrelDetLayer*>(l));
  else
    reachableFL .push_back(dynamic_cast<ForwardDetLayer*>(l));
}


DetLayer * CfgNavigationSchool::layer(std::string & lname){
  std::string part = lname.substr(0,3);
  unsigned int idLayer = atoi(lname.substr(3,1).c_str())-1;
  bool isFwd = (lname.find("pos")!=std::string::npos);
  LogDebug("CfgNavigationSchool")<<"part: "<<part<<"\n idLayer: "<<idLayer<<" is: "<<(isFwd?"isFwd":"isBwd");
  if (part == "TOB") return theTracker->tobLayers()[idLayer];
  else if (part == "TIB")  return theTracker->tibLayers()[idLayer];
  else if (part == "TID") return (isFwd?theTracker->posTidLayers()[idLayer]:theTracker->negTidLayers()[idLayer]);
  else if (part == "TEC") return (isFwd?theTracker->posTecLayers()[idLayer]:theTracker->negTecLayers()[idLayer]);
  else if (part == "PXB") return theTracker->pixelBarrelLayers()[idLayer];
  else if (part == "PXF") return (isFwd?theTracker->posPixelForwardLayers()[idLayer]:theTracker->negPixelForwardLayers()[idLayer]);
  
  edm::LogError("CfgNavigationSchool")<<"layer specification: "<<lname<<" not understood. returning a null pointer.";
  return 0;//and crash

}
