#ifndef TrackingTools_Record_DetIdAssociatorRecord_h
#define TrackingTools_Record_DetIdAssociatorRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

#include "boost/mpl/vector.hpp"

class DetIdAssociatorRecord : public edm::eventsetup::DependentRecordImplementation<DetIdAssociatorRecord,
  boost::mpl::vector<CaloGeometryRecord, GlobalTrackingGeometryRecord, CSCBadChambersRcd> > {};

#endif
