#ifndef RecoTracker_TkNavigation_NavigationSchoolESProducer_h
#define RecoTracker_TkNavigation_NavigationSchoolESProducer_h

#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/TkNavigation/interface/NavigationSchoolFactory.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//
// class decleration
//

class NavigationSchoolESProducer : public edm::ESProducer {
public:
  NavigationSchoolESProducer(const edm::ParameterSet&);
  ~NavigationSchoolESProducer();
  
  typedef boost::shared_ptr<NavigationSchool> ReturnType;

  virtual ReturnType produce(const NavigationSchoolRecord&);
 protected:
  // ----------member data ---------------------------
  edm::ParameterSet theNavigationPSet;
  std::string theNavigationSchoolName;
  boost::shared_ptr<NavigationSchool> theNavigationSchool ;
};

#endif
