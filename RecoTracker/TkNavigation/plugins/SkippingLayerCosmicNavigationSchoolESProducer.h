#ifndef RecoTracker_TkNavigation_SkippingLayerNavigationSchoolESProducer_h
#define RecoTracker_TkNavigation_SkippingLayerNavigationSchoolESProducer_h

#include "RecoTracker/TkNavigation/plugins/NavigationSchoolESProducer.h"

//
// class decleration
//

class SkippingLayerCosmicNavigationSchoolESProducer : public NavigationSchoolESProducer {
 public:
  SkippingLayerCosmicNavigationSchoolESProducer(const edm::ParameterSet& iConfig)
    : NavigationSchoolESProducer(iConfig){}
  ~SkippingLayerCosmicNavigationSchoolESProducer(){}
  
  ReturnType produce(const NavigationSchoolRecord&);
};

#endif
