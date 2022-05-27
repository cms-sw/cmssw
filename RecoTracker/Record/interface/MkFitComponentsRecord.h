#ifndef RecoTracker_Record_MkFitComponentsRecord_h
#define RecoTracker_Record_MkFitComponentsRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"


class MkFitComponentsRecord
    : public edm::eventsetup::DependentRecordImplementation<MkFitComponentsRecord,
                                                            edm::mpl::Vector<TrackerRecoGeometryRecord,
                                                                             IdealMagneticFieldRecord> > {};
#endif
