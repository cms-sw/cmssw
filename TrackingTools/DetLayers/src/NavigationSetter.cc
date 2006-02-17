#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigableLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"


NavigationSetter::NavigationSetter( const NavigationSchool& school) 
{
  StateType newState = school.navigableLayers();
  saveState( newState);
  setState( newState);
}

NavigationSetter::~NavigationSetter() 
{
  setState( theSavedState);
}


void NavigationSetter::saveState( const StateType& newState) {
  for ( StateType::const_iterator i = newState.begin(); i != newState.end(); i++) {
    if ( *i != 0) theSavedState.push_back( (**i).detLayer()->navigableLayer());
  }
}

void NavigationSetter::setState( const StateType& newState) {
  for ( StateType::const_iterator i = newState.begin(); i != newState.end(); i++) {
    if ( *i != 0) (**i).detLayer()->setNavigableLayer(*i);
  }
}

