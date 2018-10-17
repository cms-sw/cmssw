#include "SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

/** Concrete navigation school for the Tracker, connecting disks only for traversing tracks : moslty beam halo muon 
 */
class dso_hidden BeamHaloNavigationSchool final : public SimpleNavigationSchool {
public:
  
  BeamHaloNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
  ~BeamHaloNavigationSchool() override{ cleanMemory();}

 protected:
  //addon to SimpleNavigationSchool
  void linkOtherEndLayers( SymmetricLayerFinder& symFinder);
  void addInward(const DetLayer * det, const FDLC& news);
  void addInward(const DetLayer * det, const ForwardDetLayer * newF);
  void establishInverseRelations() override;
  FDLC reachableFromHorizontal();
};


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "SymmetricLayerFinder.h"
#include "SimpleBarrelNavigableLayer.h"
#include "SimpleForwardNavigableLayer.h"
#include "SimpleNavigableLayer.h"

using namespace std;

BeamHaloNavigationSchool::BeamHaloNavigationSchool(const GeometricSearchTracker* theInputTracker,
					       const MagneticField* field) 
{
  edm::LogInfo("BeamHaloNavigationSchool")<<"*********Running BeamHaloNavigationSchool *********";
  theBarrelLength = 0;theField = field; theTracker = theInputTracker;
  theAllDetLayersInSystem=&theInputTracker->allLayers();
  theAllNavigableLayer.resize(theInputTracker->allLayers().size(),nullptr);




  // Get barrel layers
  /*sideways does not need barrels*/
  /*  vector<BarrelDetLayer*> blc = theTracker->barrelLayers(); 
      for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
      theBarrelLayers.push_back( (*i) );
      }*/

  // get forward layers
  for( auto const& l : theTracker->forwardLayers()) {
    theForwardLayers.push_back(l);
  }
  
  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
			 not1(DetBelowZ(0)));
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
  theRightLayers = FDLC( middle, theForwardLayers.end());
  
  SymmetricLayerFinder symFinder( theForwardLayers);

  // only work on positive Z side; negative by mirror symmetry later
  /*sideways does not need barrels*/
  //  linkBarrelLayers( symFinder);

  linkForwardLayers( symFinder);
 
  setState(navigableLayers());

  LogDebug("BeamHaloNavigationSchool")<<"inverse relation";
  establishInverseRelations();


  //add the necessary inward links to end caps
  LogDebug("BeamHaloNavigationSchool")<<"linkOtherEndLayer";

  linkOtherEndLayers( symFinder);

  //set checkCrossing = false to all layers
  SimpleNavigationSchool::StateType allLayers=navigableLayers();
  SimpleNavigationSchool::StateType::iterator layerIt=allLayers.begin();
  SimpleNavigationSchool::StateType::iterator layerIt_end=allLayers.end();
  for (;layerIt!=layerIt_end;++layerIt)
    {
      //convert to SimpleNavigableLayer
      SimpleNavigableLayer* snl=dynamic_cast<SimpleNavigableLayer*>(*layerIt);
      if (!snl){
	edm::LogError("BeamHaloNavigationSchool")<<"navigable layer not casting to simplenavigablelayer.";
	continue;}
      snl->setCheckCrossingSide(false);
    }

}

void BeamHaloNavigationSchool::establishInverseRelations() {

  // find for each layer which are the barrel and forward
  // layers that point to it
  typedef map<const DetLayer*, vector<BarrelDetLayer const*>, less<const DetLayer*> > BarrelMapType;
  typedef map<const DetLayer*, vector<ForwardDetLayer const*>, less<const DetLayer*> > ForwardMapType;


  BarrelMapType reachedBarrelLayersMap;
  ForwardMapType reachedForwardLayersMap;

  for ( BDLI bli = theBarrelLayers.begin();
        bli!=theBarrelLayers.end(); bli++) {
    DLC reachedLC = nextLayers(**bli, insideOut);
    for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
      reachedBarrelLayersMap[*i].push_back( *bli);
    }
  }

  for ( FDLI fli = theForwardLayers.begin();
        fli!=theForwardLayers.end(); fli++) {
    DLC reachedLC = nextLayers(**fli, insideOut);
    for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
      reachedForwardLayersMap[*i].push_back( *fli);
    }
  }


  for ( auto const i : theTracker->allLayers()) {
    SimpleNavigableLayer* navigableLayer =
     dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[i->seqNum()]);
    if (!navigableLayer) {edm::LogInfo("BeamHaloNavigationSchool")<<"a detlayer does not have a navigable layer, which is normal in beam halo navigation.";}
    if (navigableLayer){navigableLayer->setInwardLinks( reachedBarrelLayersMap[i],reachedForwardLayersMap[i], TkLayerLess(outsideIn, i) );}
  }

}


void BeamHaloNavigationSchool::
linkOtherEndLayers(  SymmetricLayerFinder& symFinder){

  LogDebug("BeamHaloNavigationSchool")<<"reachable from horizontal";
  //generally, on the right side, what are the forward layers reachable from the horizontal
  FDLC reachableFL= reachableFromHorizontal();

  //even simpler navigation from end to end.
  //for each of them
  for (FDLI fl=reachableFL.begin();fl!=reachableFL.end();fl++)
    {
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from right";
      //link it inward to the mirror reachable from horizontal
      addInward(static_cast<DetLayer const*>(*fl),symFinder.mirror(*fl));
      
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from mirror of right (left?)";
      addInward(static_cast<DetLayer const*>(symFinder.mirror(*fl)),*fl);
    }




}

void BeamHaloNavigationSchool::
addInward(const DetLayer * det, const ForwardDetLayer * newF){
  //get the navigable layer for this DetLayer
  SimpleNavigableLayer* navigableLayer =
    dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(det)->seqNum()]);

  LogDebug("BeamHaloNavigationSchool")<<"retreive the nextlayer outsidein";
  //get the inward reachable layers.
  DLC inwardsLayers(navigableLayer->nextLayers(outsideIn));
  
  LogDebug("BeamHaloNavigationSchool")<<"split them barrel/forward";
  // split barrel and forward layers
  BDLC inwardsBarrel;
  FDLC inwardsForward;
  for ( DLC::iterator dli=inwardsLayers.begin();dli!=inwardsLayers.end();dli++)
    {
      if ((**dli).location()==GeomDetEnumerators::barrel)
        inwardsBarrel.push_back(static_cast<const BarrelDetLayer*>(*dli));
      else
        inwardsForward.push_back(static_cast<const ForwardDetLayer*>(*dli));
    }
  LogDebug("BeamHaloNavigationSchool")<<"add the new ones";
  //add the other forward layers provided
  inwardsForward.push_back(newF);

  LogDebug("BeamHaloNavigationSchool")<<"no duplicate please";
  sort(inwardsForward.begin(),inwardsForward.end()); //if you don't sort, unique will not work
  //  FDLI read = inwardsForward.begin();
  //  std::stringstream showMe;
  //  for (;read !=inwardsForward.end();++read)  showMe<<" layer p: "<<*read<<"\n";
  //  LogDebug("BeamHaloNavigationSchool")<<"list of layer pointers: \n"<<showMe.str();

  FDLI new_end =unique(inwardsForward.begin(),inwardsForward.end());
  //  if (new_end!=inwardsForward.end()) LogDebug("BeamHaloNavigationSchool")<<"removing duplicates here";
  inwardsForward.erase(new_end,inwardsForward.end());

  LogDebug("BeamHaloNavigationSchool")<<"set back the inward links (no duplicate)";
  //  set them back to the navigable layer
  navigableLayer->setInwardLinks( inwardsBarrel, inwardsForward, TkLayerLess(outsideIn, det));
}

void BeamHaloNavigationSchool::
addInward(const DetLayer * det, const FDLC& news){
  //get the navigable layer for this DetLayer
  SimpleNavigableLayer* navigableLayer =
    dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(det)->seqNum()]);

  LogDebug("BeamHaloNavigationSchool")<<"retreive the nextlayer outsidein";
  //get the inward reachable layers.
  DLC inwardsLayers(navigableLayer->nextLayers(outsideIn));

  LogDebug("BeamHaloNavigationSchool")<<"split them barrel/forward";
  // split barrel and forward layers
  BDLC inwardsBarrel;
  FDLC inwardsForward;
  for ( DLC::iterator dli=inwardsLayers.begin();dli!=inwardsLayers.end();dli++)
    {
      if ((**dli).location()==GeomDetEnumerators::barrel)
	inwardsBarrel.push_back(static_cast<const BarrelDetLayer*>(*dli));
      else
	inwardsForward.push_back(static_cast<const ForwardDetLayer*>(*dli));
    }
  
  LogDebug("BeamHaloNavigationSchool")<<"add the new ones";
  //add the other forward layers provided
  inwardsForward.insert( inwardsForward.end(), news.begin(), news.end());

  LogDebug("BeamHaloNavigationSchool")<<"no duplicate please";
  FDLI new_end =unique(inwardsForward.begin(),inwardsForward.end());
  inwardsForward.erase(new_end,inwardsForward.end());

  LogDebug("BeamHaloNavigationSchool")<<"set back the inward links (no duplicate)";
  //  set them back to the navigable layer
  navigableLayer->setInwardLinks( inwardsBarrel, inwardsForward, TkLayerLess(outsideIn, det));
}

BeamHaloNavigationSchool::FDLC
BeamHaloNavigationSchool::reachableFromHorizontal()
{    
  //determine which is the list of forward layers that can be reached from inside-out
  //at horizontal direction

  FDLC myRightLayers( theRightLayers);
  FDLI begin = myRightLayers.begin();
  FDLI end   = myRightLayers.end();

  //sort along Z to be sure
  sort(begin, end, isDetLessZ);

  FDLC reachableFL;

  begin = myRightLayers.begin();
  end   = myRightLayers.end();

  //the first one is always reachable
  reachableFL.push_back(*begin);
  FDLI current = begin;
  for (FDLI i = begin+1; i!= end; i++)
    {
      //is the previous layer NOT masking this one
      //inner radius smaller OR outer radius bigger
      if ((**i).specificSurface().innerRadius() < (**current).specificSurface().innerRadius() ||
	  (**i).specificSurface().outerRadius() > (**current).specificSurface().outerRadius())
	{	  //not masked
	  reachableFL.push_back(*i);
	  current=i;
	}
    }
  return reachableFL;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

 
#include "NavigationSchoolFactory.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, BeamHaloNavigationSchool, "BeamHaloNavigationSchool");

