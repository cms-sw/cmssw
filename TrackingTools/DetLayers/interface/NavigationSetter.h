#ifndef DetLayers_NavigationSetter_H
#define DetLayers_NavigationSetter_H

#include <vector>

class NavigableLayer;
class NavigationSchool;
class DetLayer;

/** This class sets the navigation state given by the NavigationSchool,
 *  and saves the previous state for the affected layers.
 *  It must be instatiated as a local variable in the method that
 *  needs the navigation of the school, before using the navigation
 *  methods. The constructor saves the current state and sets the new one,
 *  and the destructod restores the previous state.
 *  This allows different reconstruction algorithms to use different 
 *  navigations in the same set of DetLayers.
 */

class NavigationSetter {
public:

  typedef std::vector<NavigableLayer*>   StateType;

  NavigationSetter( const NavigationSchool&);

  ~NavigationSetter();

private:

  const NavigationSchool& theNav;

  StateType theSavedState;

  void saveState();
  void setState( const StateType&); 
  void cleanState();

};

#endif // NavigationSetter_H
