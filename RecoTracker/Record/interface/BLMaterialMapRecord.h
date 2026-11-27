#ifndef RecoTracker_Record_BLMaterialMapRecord_h
#define RecoTracker_Record_BLMaterialMapRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record of the material map of the BrokenLine fit (interface/BLMaterialMap.h), keyed to the ideal
// geometry.
class BLMaterialMapRecord
    : public edm::eventsetup::DependentRecordImplementation<BLMaterialMapRecord, edm::mpl::Vector<IdealGeometryRecord> > {
};

#endif  // RecoTracker_Record_BLMaterialMapRecord_h
