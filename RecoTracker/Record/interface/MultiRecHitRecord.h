#ifndef RecoLocalTracker_Records_MultiRecHitRecord_h
#define RecoLocalTracker_Records_MultiRecHitRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "boost/mpl/vector.hpp"

class MultiRecHitRecord
    : public edm::eventsetup::DependentRecordImplementation<
          MultiRecHitRecord,
          boost::mpl::vector<TrackerDigiGeometryRecord, TransientRecHitRecord,
                             CkfComponentsRecord>> {};
#endif
