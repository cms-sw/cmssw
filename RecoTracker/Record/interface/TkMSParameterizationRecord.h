#ifndef RecoTracker_Record_TkMSParameterizationRecord_h
#define RecoTracker_Record_TkMSParameterizationRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"


#include "boost/mpl/vector.hpp"


class TkMSParameterizationRecord : public edm::eventsetup::DependentRecordImplementation<TkMSParameterizationRecord,
  boost::mpl::vector<TrackingComponentsRecord,NavigationSchoolRecord> > {};

#endif 

