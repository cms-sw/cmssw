#ifndef TrackingTools_Record_TransientRecHitRecord_h
#define TrackingTools_Record_TransientRecHitRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"

#include "boost/mpl/vector.hpp"

class TransientRecHitRecord
    : public edm::eventsetup::DependentRecordImplementation<
          TransientRecHitRecord,
          boost::mpl::vector<CaloGeometryRecord, TrackerDigiGeometryRecord,
                             TkStripCPERecord, TkPixelCPERecord,
                             GlobalTrackingGeometryRecord>> {};
#endif
