#include <cstddef>
#include <sstream>
#include <cmath>

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// HGCal imports
#include "SimCalorimetry/HGCalSimProducers/interface/alpaka/HGCalDigitizerAlgorithms.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // @short this kernel takes care of adding noise on each bunch crossing
  struct HGCalECONDAlgosKernel_fillCM {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalECONDPacketInfoDevice::View econdInfo,
                                  HGCalMappingModuleParamDevice::ConstView moduleInfo,
                                  const uint64_t seed) const {
      // one engine per thread (or per work item)
      const uint32_t globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      auto engine = alpaka::rand::engine::createDefault(acc, seed + globalThreadIdx, 0);
      auto normal = alpaka::rand::distribution::createNormalReal<float>(acc);

      // dispatch operation
      for (auto idx : uniform_elements(acc, econdInfo.metadata().size())) {
        auto econd = econdInfo[idx];

        //placeholder - in practice we should get the gain associated with the module and then throw the typical expected noise for 47pF with that gain
        // in addition there would be no need to generate 12 values just the ones corresponding to 2*nErx
        for (size_t i = 0; i < 12; i++)
          for (size_t j = 0; j < 2; j++) {
            const float noiseVal = normal(engine);
            econd.cm()(i, j) = noiseVal;
          }

      }  //end grid loop
    }
  };

  // @short apply the sequence of digitization kernels
  HGCalECONDPacketInfoDevice HGCalECONDAlgorithms::init(Queue& queue,
                                                        HGCalAgeingCondsDevice const& ageConds,
                                                        HGCalMappingModuleParamDevice const& moduleInfo,
                                                        const uint64_t seed) const {
    LogDebug("HGCalECONDAlgorithms") << "Start of ECON-D info" << std::endl;
    auto const necons = moduleInfo.view().metadata().size();
    HGCalECONDPacketInfoDevice device_econds(queue, necons);

    // number of items per group and division in groups to cover the whole problem
    uint32_t items = n_threads_;
    uint32_t groups = divide_up_by(necons, items);

    // generate the working grid
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalECONDAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    //execute digitization kernels
    alpaka::exec<Acc1D>(
        queue, grid, HGCalECONDAlgosKernel_fillCM{}, device_econds.view(), moduleInfo.const_view(), seed);

    return device_econds;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
