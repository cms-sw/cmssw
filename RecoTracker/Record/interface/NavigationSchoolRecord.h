#ifndef RecoTracker_Record_NavigationSchoolRecord_h
#define RecoTracker_Record_NavigationSchoolRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class NavigationSchoolRecord : public edm::eventsetup::DependentRecordImplementation<
                                   NavigationSchoolRecord,
                                   edm::mpl::Vector<IdealMagneticFieldRecord, TrackerRecoGeometryRecord> > {};

#endif
