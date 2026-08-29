#ifndef RecoTracker_Record_BLBFieldMapRecord_h
#define RecoTracker_Record_BLBFieldMapRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record for the BL-fit axial-field shape map s(z)=B(z)/B(0), keyed to the MagneticField IOV.
class BLBFieldMapRecord
    : public edm::eventsetup::DependentRecordImplementation<BLBFieldMapRecord,
                                                            edm::mpl::Vector<IdealMagneticFieldRecord> > {};

#endif  // RecoTracker_Record_BLBFieldMapRecord_h
