#include "RecoTracker/PixelSeeding/interface/IntermediateHitTriplets.h"
#include "FWCore/Utilities/interface/Exception.h"

IntermediateHitTriplets::IntermediateHitTriplets(const IntermediateHitTriplets& rh) {
  throw cms::Exception("Not Implemented") << "The copy constructor of IntermediateHitTriplets should never be called. "
                                             "The function exists only to make ROOT dictionary generation happy.";
}
