#ifndef RecoTracker_Record_BLBFieldMapRecord_h
#define RecoTracker_Record_BLBFieldMapRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record for the BL-fit normalized (Bz,Br) r-z field map (interface/BLBFieldMap.h), sampled from
// the MagneticField product and keyed to its IOV.
class BLBFieldMapRecord
    : public edm::eventsetup::DependentRecordImplementation<BLBFieldMapRecord,
                                                            edm::mpl::Vector<IdealMagneticFieldRecord> > {};

#endif  // RecoTracker_Record_BLBFieldMapRecord_h
