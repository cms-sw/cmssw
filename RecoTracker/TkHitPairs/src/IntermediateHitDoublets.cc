#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "FWCore/Utilities/interface/Exception.h"

IntermediateHitDoublets::IntermediateHitDoublets(const IntermediateHitDoublets& rh) {
  throw cms::Exception("Not Implemented") << "The copy constructor of IntermediateHitDoublets should never be called. "
                                             "The function exists only to make ROOT dictionary generation happy.";
}
