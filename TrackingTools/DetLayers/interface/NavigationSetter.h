#ifndef DetLayers_NavigationSetter_H
#define DetLayers_NavigationSetter_H

#include <vector>

class NavigableLayer;
class NavigationSchool;

/** This class sets the navigation state given by the NavigationSchool,
 *  and saves the previous state for the affected layers.
 *  It must be instatiated as a local variable in the method that
 *  needs the navigation of the school, before using the navigation
 *  methods. The constructor saves the current state and sets the new one,
 *  and the destructod restores the previous state.
 *  This allows different reconstruction algorithms to use different 
 *  navigations in the same set of DetLayers.
 */

using namespace std;
class NavigationSetter {
public:

  typedef vector<NavigableLayer*>   StateType;

  NavigationSetter( const NavigationSchool&);

  ~NavigationSetter();

private:

  StateType theSavedState;

  void saveState( const StateType& newState);
  void setState( const StateType&); 

};

#endif // NavigationSetter_H
