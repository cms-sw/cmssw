#ifndef TrackingTools_RecoGeometry_RecoGeometryRecord_h
#define TrackingTools_RecoGeometry_RecoGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
//#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/mpl/vector.hpp"


class RecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<RecoGeometryRecord,
  boost::mpl::vector<TrackerRecoGeometryRecord,MuonRecoGeometryRecord
			   //,NavigationSchoolRecord,
			   //IdealMagneticFieldRecord
  > > {};

#endif 
