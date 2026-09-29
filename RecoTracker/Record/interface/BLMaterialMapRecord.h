#ifndef RecoTracker_Record_BLMaterialMapRecord_h
#define RecoTracker_Record_BLMaterialMapRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record for the BL-fit Geant4 material map rho(r,z), keyed to the tracker geometry IOV.
class BLMaterialMapRecord
    : public edm::eventsetup::DependentRecordImplementation<BLMaterialMapRecord,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord> > {};

#endif  // RecoTracker_Record_BLMaterialMapRecord_h
