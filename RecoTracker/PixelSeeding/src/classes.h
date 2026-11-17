#include "RecoTracker/PixelSeeding/interface/IntermediateHitTriplets.h"
#include "DataFormats/Common/interface/Wrapper.h"

// Per-built-triplet training-dataset host SoA (CA_TRIPLET_DUMP product). Type-info dictionary only
// (no memory); needed so the device 'Triplet' product can be transcribed to a host collection and
// read by TripletFeaturesTableProducer. Unconditional: harmless when CA_TRIPLET_DUMP is off.
#include "RecoTracker/PixelSeeding/interface/TripletDumpHost.h"

#include <vector>

namespace RecoPixelVertexing_PixelTriplets {
  struct dictionary {
    IntermediateHitTriplets iht;
    edm::Wrapper<IntermediateHitTriplets> wiht;
  };
}  // namespace RecoPixelVertexing_PixelTriplets
