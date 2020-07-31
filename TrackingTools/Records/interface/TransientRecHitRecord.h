#ifndef TrackingTools_Record_TransientRecHitRecord_h
#define TrackingTools_Record_TransientRecHitRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class TransientRecHitRecord
    : public edm::eventsetup::DependentRecordImplementation<TransientRecHitRecord,
                                                            edm::mpl::Vector<CaloGeometryRecord,
                                                                             TrackerDigiGeometryRecord,
                                                                             TkStripCPERecord,
                                                                             TkPixelCPERecord,
                                                                             TkPhase2OTCPERecord,
                                                                             GlobalTrackingGeometryRecord> > {};
#endif
