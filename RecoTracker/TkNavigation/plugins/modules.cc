#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
 
#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/CosmicNavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/BeamHaloNavigationSchool.h"
 
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, SimpleNavigationSchool, "SimpleNavigationSchool");
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, CosmicNavigationSchool, "CosmicNavigationSchool");
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, BeamHaloNavigationSchool, "BeamHaloNavigationSchool");
