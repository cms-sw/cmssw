#ifndef TkNavigation_TkNavigationSchool_H
#define TkNavigation_TkNavigationSchool_H


#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

class MagneticField;

/** Concrete navigation school for the Tracker
 */

class TkNavigationSchool : public NavigationSchool {
public:
  
  TkNavigationSchool(const GeometricSearchTracker* tracker,
	             const MagneticField* field) :
                     theField(field), theTracker(tracker){}
 
  const MagneticField & field() const {return *theField;}
  const GeometricSearchTracker & searchTracker() const { return *theTracker;}

protected:

  const MagneticField* theField;
  const GeometricSearchTracker* theTracker;

};

#endif

