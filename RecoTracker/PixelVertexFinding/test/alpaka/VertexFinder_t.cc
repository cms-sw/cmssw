#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"

#include "RecoTracker/PixelVertexFinding/interface/PixelVertexWorkSpaceLayout.h"
#include "RecoTracker/PixelVertexFinding/plugins/PixelVertexWorkSpaceSoAHostAlpaka.h"
#include "RecoTracker/PixelVertexFinding/plugins/alpaka/PixelVertexWorkSpaceSoADeviceAlpaka.h"

using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace vertexfinder_t {
    void runKernels(Queue& queue);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  vertexfinder_t::runKernels(queue);
  return 0;
}
