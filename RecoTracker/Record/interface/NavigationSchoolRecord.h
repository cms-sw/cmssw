#ifndef RecoTracker_Record_NavigationSchoolRecord_h
#define RecoTracker_Record_NavigationSchoolRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"


#include "boost/mpl/vector.hpp"


class NavigationSchoolRecord : public edm::eventsetup::DependentRecordImplementation<NavigationSchoolRecord,
  boost::mpl::vector<IdealMagneticFieldRecord, TrackerRecoGeometryRecord > > {};

#endif 

