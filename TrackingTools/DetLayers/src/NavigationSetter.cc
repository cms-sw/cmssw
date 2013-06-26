#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

NavigationSetter::NavigationSetter( const NavigationSchool& school) 
  : theNav(school)
{
  //save the current state from all layer in the system which is being set: allow to set it back
  saveState();

  //remove any links from the detlayer in the system: allow partial navigation (BeamHalo)
  cleanState();

  //set the current navigation
  setState(school.navigableLayers());
}

NavigationSetter::~NavigationSetter() 
{
  //remove any link from the detlyaer in the system which has been set
  cleanState();
  /*  
      LogDebug("NavigationSetter")<<"NavigationSchool settings are totally reset in the destructor of NavigationSetter.\n"
      <<"this is the designed behavior. If you do not get track reconstruction, please refer to\n"
      <<"https://twiki.cern.ch/twiki/bin/view/CMS/NavigationSchool \n"
      <<"to properly use the NavigationSetter.";
  */

  //restore the previous layer DetLayer settings within the system which has been set.
  setState( theSavedState);
}


void NavigationSetter::saveState() {
  //remember all navigable layers from all layer in the system.
  std::vector<DetLayer*>::const_iterator i = theNav.allLayersInSystem().begin();
  std::vector<DetLayer*>::const_iterator end = theNav.allLayersInSystem().end();
  for (; i != end; ++i){ 
    if (*i !=0) theSavedState.push_back( (*i)->navigableLayer());
  }
}

void NavigationSetter::cleanState(){
  //set no navigable layer to all detlayers in the system
  std::vector<DetLayer*>::const_iterator i = theNav.allLayersInSystem().begin();
  std::vector<DetLayer*>::const_iterator end = theNav.allLayersInSystem().end();
  for (; i != end; ++i){
    if (*i !=0) (*i)->setNavigableLayer(0);
  }
}

void NavigationSetter::setState( const StateType& newState) {
  //set DetLayer->NavigableLayer link from navigable layer in given state
  StateType::const_iterator i = newState.begin();
  StateType::const_iterator end = newState.end();
  for (; i != end; i++) {
    if ( *i != 0) (**i).detLayer()->setNavigableLayer(*i);
  }
}

