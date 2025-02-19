#ifndef RecoTracker_TkNavigation_NavigationSchoolFactory_h
#define RecoTracker_TkNavigation_NavigationSchoolFactory_h

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/PluginManager/interface/PluginFactory.h" 

typedef edmplugin::PluginFactory<NavigationSchool *(const GeometricSearchTracker* theTracker,const MagneticField* field)> NavigationSchoolFactory;

#endif 
