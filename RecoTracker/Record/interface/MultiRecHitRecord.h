#ifndef RecoLocalTracker_Records_MultiRecHitRecord_h
#define RecoLocalTracker_Records_MultiRecHitRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include <boost/mp11/list.hpp>

class MultiRecHitRecord
    : public edm::eventsetup::DependentRecordImplementation<
          MultiRecHitRecord,
          boost::mp11::mp_list<TrackerDigiGeometryRecord, TransientRecHitRecord, CkfComponentsRecord> > {};
#endif
