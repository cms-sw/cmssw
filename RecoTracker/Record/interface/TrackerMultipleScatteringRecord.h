#ifndef RecoTracker_Record_TrackerMultipleScatteringRecord_h
#define RecoTracker_Record_TrackerMultipleScatteringRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TrackerMultipleScatteringRecord : public edm::eventsetup::DependentRecordImplementation<
                                            TrackerMultipleScatteringRecord,
                                            edm::mpl::Vector<IdealMagneticFieldRecord, TrackerRecoGeometryRecord> > {};

#endif
