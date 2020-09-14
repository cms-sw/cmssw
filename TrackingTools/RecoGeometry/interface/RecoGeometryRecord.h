#ifndef TrackingTools_RecoGeometry_RecoGeometryRecord_h
#define TrackingTools_RecoGeometry_RecoGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
//#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class RecoGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                               RecoGeometryRecord,
                               edm::mpl::Vector<TrackerRecoGeometryRecord, MuonRecoGeometryRecord, MTDRecoGeometryRecord
                                                //,NavigationSchoolRecord,
                                                //IdealMagneticFieldRecord
                                                > > {};

#endif
