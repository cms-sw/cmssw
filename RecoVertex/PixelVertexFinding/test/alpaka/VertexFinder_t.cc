#include <cstdlib>
#include <iostream>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

#include "VertexFinder_t.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // run the test on all the available devices
  for (auto const& device : devices) {
    Queue queue(device);
    vertexfinder_t::runKernels(queue);
  }

  return EXIT_SUCCESS;
}
