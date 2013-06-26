#ifndef RecoTracker_Record_TransientTrackRecord_h
#define RecoTracker_Record_TransientTrackRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"


#include "boost/mpl/vector.hpp"

class TransientTrackRecord : public edm::eventsetup::DependentRecordImplementation<TransientTrackRecord,
  boost::mpl::vector<IdealMagneticFieldRecord, GlobalTrackingGeometryRecord> > {};
#endif 

