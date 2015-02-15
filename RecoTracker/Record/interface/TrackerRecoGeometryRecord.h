#ifndef RecoTracker_Record_TrackerRecoGeometryRecord_h
#define RecoTracker_Record_TrackerRecoGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "boost/mpl/vector.hpp"


class TrackerRecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<TrackerRecoGeometryRecord,
  boost::mpl::vector<TrackerTopologyRcd,TrackerDigiGeometryRecord> > {};

#endif 

