#ifndef RecoLocalTracker_Records_MultiRecHitRecord_h
#define RecoLocalTracker_Records_MultiRecHitRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class MultiRecHitRecord
    : public edm::eventsetup::DependentRecordImplementation<
          MultiRecHitRecord,
          edm::mpl::Vector<TrackerDigiGeometryRecord, TransientRecHitRecord, CkfComponentsRecord> > {};
#endif
