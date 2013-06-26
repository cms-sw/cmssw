#ifndef RecoTracker_Record_TrackingComponentsRecord_h
#define RecoTracker_Record_TrackingComponentsRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"                

#include "boost/mpl/vector.hpp"


class  TrackingComponentsRecord: public edm::eventsetup::DependentRecordImplementation<TrackingComponentsRecord,
  boost::mpl::vector<IdealMagneticFieldRecord, GlobalTrackingGeometryRecord> > {};

#endif 

