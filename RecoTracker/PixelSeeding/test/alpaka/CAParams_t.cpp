/**
   Simple test for the reco::CAParamsSoA data structure
   which inherits from Portable{Host}Collection.
 */

#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/PixelSeeding/interface/CAParamsHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAParamsSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CAParams_t.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;
// Run 3 values, used for testing
constexpr uint32_t n_layers = 15;
constexpr uint32_t n_pairs = 50;
constexpr uint32_t n_regions = 5;

int main() {
  // Get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device
  for (const auto& device : devices) {
    Queue queue(device);

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate vertices on device. PortableCollection allocates
      // SoA on device automatically.
      CAParamsSoACollection ca_params_d({{n_layers,n_pairs}}, queue);
      // testParamsSoA::runKernels(ca_params_d.view(), ca_params_d.view<reco::CACellsSoA>(), ca_params_d.view<reco::CARegionsSoA>(), queue);

//       // If the device is actually the host, use the collection as-is.
//       // Otherwise, copy the data from the device to the host.
//       ZVertexHost zvertex_h;
// #ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
//       zvertex_h = std::move(zvertex_d);
// #else
//       zvertex_h = cms::alpakatools::CopyToHost<ZPixelSeedingCollection>::copyAsync(queue, zvertex_d);
// #endif
//       alpaka::wait(queue);
//       std::cout << zvertex_h.view().metadata().size() << std::endl;

//       // Print results
//       std::cout << "idv\t"
//                 << "zv\t"
//                 << "wv\t"
//                 << "chi2\t"
//                 << "ptv2\t"
//                 << "ndof\t"
//                 << "sortInd\t"
//                 << "nvFinal\n";

//       auto vtx_v = zvertex_h.view<reco::ZPixelSeeding>();
//       auto trk_v = zvertex_h.view<reco::ZVertexTracksSoA>();
//       for (int i = 0; i < 10; ++i) {
//         auto vi = vtx_v[i];
//         auto ti = trk_v[i];
//         std::cout << (int)ti.idv() << "\t" << vi.zv() << "\t" << vi.wv() << "\t" << vi.chi2() << "\t" << vi.ptv2()
//                   << "\t" << (int)ti.ndof() << "\t" << vi.sortInd() << "\t" << (int)vtx_v.nvFinal() << std::endl;
//       }
    }
  }

  return EXIT_SUCCESS;
}