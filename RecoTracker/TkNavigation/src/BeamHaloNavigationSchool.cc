#include "RecoTracker/TkNavigation/interface/BeamHaloNavigationSchool.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"
#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

using namespace std;

BeamHaloNavigationSchool::BeamHaloNavigationSchool(const GeometricSearchTracker* theInputTracker,
					       const MagneticField* field) 
{
  edm::LogInfo("BeamHaloNavigationSchool")<<"*********Running BeamHaloNavigationSchool *********";
  theBarrelLength = 0;theField = field; theTracker = theInputTracker;
  theAllDetLayersInSystem=&theInputTracker->allLayers();
  // Get barrel layers
  /*sideways does not need barrels*/
  /*  vector<BarrelDetLayer*> blc = theTracker->barrelLayers(); 
      for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
      theBarrelLayers.push_back( (*i) );
      }*/

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
  /*sideways does not need barrels*/
  //  linkBarrelLayers( symFinder);

  linkForwardLayers( symFinder);
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
    SimpleNavigableLayer* navigableLayer =
      dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
    if (!navigableLayer) {edm::LogInfo("BeamHaloNavigationSchool")<<"a detlayer does not have a navigable layer, which is normal in beam halo navigation.";}
    if (navigableLayer){navigableLayer->setInwardLinks( reachedBarrelLayersMap[*i],reachedForwardLayersMap[*i], TkLayerLess(outsideIn, (*i)) );}
  }

}


void BeamHaloNavigationSchool::
linkOtherEndLayers(  SymmetricLayerFinder& symFinder){
  NavigationSetter setter(*this);

  LogDebug("BeamHaloNavigationSchool")<<"reachable from horizontal";
  //generally, on the right side, what are the forward layers reachable from the horizontal
  FDLC reachableFL= reachableFromHorizontal();

  //even simpler navigation from end to end.
  //for each of them
  for (FDLI fl=reachableFL.begin();fl!=reachableFL.end();fl++)
    {
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from right";
      //link it inward to the mirror reachable from horizontal
      addInward((DetLayer*)*fl,symFinder.mirror(*fl));
      
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from mirror of right (left?)";
      addInward((DetLayer*)symFinder.mirror(*fl),*fl);
    }

  /* this is not enough to set reachable from each of them: too few links
     //this is enough in the end
     //for each of them
     for (FDLI fl=reachableFL.begin();fl!=reachableFL.end();fl++)
     {
     LogDebug("BeamHaloNavigationSchool")<<"adding inward from right";
     //link it inward to the mirror reachable from horizontal
     addInward((DetLayer*)*fl,symFinder.mirror(reachableFL));
     
     LogDebug("BeamHaloNavigationSchool")<<"adding inward from mirror of right (left?)";
     //do the same from the the mirrored layer to the reachable from horizontal
     addInward((DetLayer*)symFinder.mirror(*fl),reachableFL);
     }
  */


  /* what about linking every not masked layer in each group.
     except for within the same group
     
     vector<FDLC> groups splitForwardLayers();
     FDLC reachable;
     
     for ( vector<FDLC>::iterator group = groups.begin();
     group != groups.end(); group++) {
     //for each group
     
     for ( FDLI i = group->begin(); i != group->end(); i++) {
     
     for ( vector<FDLC>::iterator other_group = groups.begin();
     other_group != groups.end(); other_group++) {
     //for each other group
     
     if (other_group==group && i==group->begin()){
     //other_group is the same as group and dealing with the first layer of the group
     //link the first of each group
     reachable.push_back(other_group.front());
     continue;}
     
     //now dealing as if other_group is different than group
     for ( FDLI other_i = other_group->begin(); other_i != other_group->end(); other_i++) {
     //for each of other group
     //is the layer in the other group "masking" this one
     //inner radius smaller OR outer radius bigger
     if ((**other_i).specificSurface().innerRadius() < (**i).specificSurface().innerRadius() ||
     (**other_i).specificSurface().outerRadius() > (**i).specificSurface().outerRadius())
     {         //not masked
     reachableFL.push_back(*other_i);
     }
     }
     //do something special with the first of each group
     //do somethign special with layers in its own group
     }
     }
     } 
  */



   /* this is too much links between layers
      FDLC myRightLayers( theRightLayers);
      FDLI begin = myRightLayers.begin();
      FDLI end   = myRightLayers.end();
      
      //  for each of the right layers
      for (FDLI fl = begin;fl!=end;++fl)
      {
      //get the navigable layer for this DetLayer
      SimpleNavigableLayer* navigableLayer =
      dynamic_cast<SimpleNavigableLayer*>((*fl)->navigableLayer());
      
      LogDebug("BeamHaloNavigationSchool")<<"retreive the next layers outsidein";
      //get the OUTward reachable layers.
      DLC inwardsLayers(navigableLayer->nextLayers(insideOut));

      //what is reachable horizontaly
      FDLC thisReachableFL (reachableFL);

      LogDebug("BeamHaloNavigationSchool")<<"combine";
      //combine the two vector with a conversion to forward layer
      for (DLI i=inwardsLayers.begin();i!=inwardsLayers.end();++i)
      {
      ForwardDetLayer* fd=dynamic_cast<ForwardDetLayer*>(const_cast<DetLayer*>(*i));
      //	  ForwardDetLayer* fd=const_cast<ForwardDetLayer*>(*i);
      if (fd){
      //	    if (thisReachableFL.find(fd)==thisReachableFL.end())
      //	      {//no duplicate. insert it
      thisReachableFL.push_back(fd);
      //}
      }
      LogDebug("BeamHaloNavigationSchool")<<"investigate";
      }
      
      //please no duplicate !!!
      LogDebug("BeamHaloNavigationSchool")<<"no duplicate";
      FDLI new_end =unique(thisReachableFL.begin(),thisReachableFL.end());
      thisReachableFL.erase(new_end,thisReachableFL.end());

      //then set the inwards links
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from right";
      //link it inward to the mirror reachable from horizontal
      addInward((DetLayer*)*fl,symFinder.mirror(thisReachableFL));
      
      LogDebug("BeamHaloNavigationSchool")<<"adding inward from mirror of right (left?)";
      //do the same from the the mirrored layer to the reachable from horizontal
      addInward((DetLayer*)symFinder.mirror(*fl),thisReachableFL);
      }
   */


}

void BeamHaloNavigationSchool::
addInward(DetLayer * det, ForwardDetLayer * newF){
  //get the navigable layer for this DetLayer
  SimpleNavigableLayer* navigableLayer =
    dynamic_cast<SimpleNavigableLayer*>((*det).navigableLayer());

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
        inwardsBarrel.push_back((BarrelDetLayer*)*dli);
      else
        inwardsForward.push_back((ForwardDetLayer*)*dli);
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
addInward(DetLayer * det, FDLC news){
  //get the navigable layer for this DetLayer
  SimpleNavigableLayer* navigableLayer =
    dynamic_cast<SimpleNavigableLayer*>((*det).navigableLayer());

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
	inwardsBarrel.push_back((BarrelDetLayer*)*dli);	
      else
	inwardsForward.push_back((ForwardDetLayer*)*dli);
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
  sort(begin, end, DetLessZ());

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
