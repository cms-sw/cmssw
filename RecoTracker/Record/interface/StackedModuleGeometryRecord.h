#ifndef RecoTracker_Record_StackedModuleGeometryRecord_h
#define RecoTracker_Record_StackedModuleGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class StackedModuleGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                        StackedModuleGeometryRecord,
                                        edm::mpl::Vector<TrackerTopologyRcd, TrackerDigiGeometryRecord>> {};

#endif  // RecoTracker_Record_StackedModuleGeometryRecord_h
