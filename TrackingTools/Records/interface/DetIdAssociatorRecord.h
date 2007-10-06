#ifndef TrackingTools_Record_DetIdAssociatorRecord_h
#define TrackingTools_Record_DetIdAssociatorRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "boost/mpl/vector.hpp"

class DetIdAssociatorRecord : public edm::eventsetup::DependentRecordImplementation<DetIdAssociatorRecord,
  boost::mpl::vector<IdealGeometryRecord, GlobalTrackingGeometryRecord> > {};

#endif
