#ifndef RecoTracker_Record_BLBFieldMapRecord_h
#define RecoTracker_Record_BLBFieldMapRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record of the (Bz, Br) field map of the BrokenLine fit (interface/BLBFieldMap.h).
class BLBFieldMapRecord
    : public edm::eventsetup::DependentRecordImplementation<BLBFieldMapRecord,
                                                            edm::mpl::Vector<IdealMagneticFieldRecord> > {};

#endif  // RecoTracker_Record_BLBFieldMapRecord_h
