#ifndef TrackingTools_TrackFitters_TrajectoryFitterRecord_h
#define TrackingTools_TrackFitters_TrajectoryFitterRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "boost/mpl/vector.hpp"


class TrajectoryFitterRecord : public edm::eventsetup::DependentRecordImplementation<TrajectoryFitterRecord,
  boost::mpl::vector<TrackingComponentsRecord,RecoGeometryRecord> > {};

#endif 
