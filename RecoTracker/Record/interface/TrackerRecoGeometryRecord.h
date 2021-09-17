#ifndef RecoTracker_Record_TrackerRecoGeometryRecord_h
#define RecoTracker_Record_TrackerRecoGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TrackerRecoGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<TrackerRecoGeometryRecord,
                                                            edm::mpl::Vector<TrackerTopologyRcd,
                                                                             TrackerDigiGeometryRecord,
                                                                             MTDGeometryRecord,
                                                                             MTDTopologyRcd,
                                                                             MTDDigiGeometryRecord> > {};

#endif
