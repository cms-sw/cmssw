#ifndef RecoTracker_Record_TrackerRecoGeometryRecord_h
#define RecoTracker_Record_TrackerRecoGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <boost/mp11/list.hpp>

class TrackerRecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                      TrackerRecoGeometryRecord,
                                      boost::mp11::mp_list<TrackerTopologyRcd, TrackerDigiGeometryRecord> > {};

#endif
