#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

 
#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/CosmicNavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/BeamHaloNavigationSchool.h"
 
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, SimpleNavigationSchool, "SimpleNavigationSchool");
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, CosmicNavigationSchool, "CosmicNavigationSchool");
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, BeamHaloNavigationSchool, "BeamHaloNavigationSchool");

#include "RecoTracker/TkNavigation/plugins/NavigationSchoolESProducer.h"
#include "RecoTracker/TkNavigation/plugins/SkippingLayerCosmicNavigationSchoolESProducer.h"
#include "RecoTracker/TkNavigation/plugins/CfgNavigationSchoolESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(NavigationSchoolESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(SkippingLayerCosmicNavigationSchoolESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(CfgNavigationSchoolESProducer);
