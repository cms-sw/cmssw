#ifndef RecoTracker_TkNavigation_CfgNavigationSchoolESProducer_H
#define RecoTracker_TkNavigation_CfgNavigationSchoolESProducer_H

#include "RecoTracker/TkNavigation/plugins/NavigationSchoolESProducer.h"

//
// class decleration
//

class CfgNavigationSchoolESProducer : public NavigationSchoolESProducer {
 public:
  CfgNavigationSchoolESProducer(const edm::ParameterSet& iConfig)
    : NavigationSchoolESProducer(iConfig){}
  ~CfgNavigationSchoolESProducer(){}
  
  ReturnType produce(const NavigationSchoolRecord&);
};

#endif
