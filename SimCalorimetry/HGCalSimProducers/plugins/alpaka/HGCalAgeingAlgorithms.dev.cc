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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // @short this kernel takes care of finalizing the Geant4 energy response and assigning the final energy deposit and average time estimates
  struct HGCalAgeingKernel_finalizeResponse {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalRealisticSimHitDevice::View g4Hits,
                                  HGCalAgeingCondsDevice::ConstView conds,
                                  HGCalDenseIndexInfoDevice::ConstView denseIndex,
                                  HGCalMappingCellParamDevice::ConstView cellInfo) const {
      //Si: E[keV] - > q [fC] 1000 eV/ 3.62 (eV per e) / 6.24150934e3 (e per fC)
      constexpr float keV2fC = 0.044259;
      //SiPM : this is a reference value for a 3 mm^2 tile
      constexpr float keV2MIP = 1. / 675.0;

      // helper for common operations
      auto scaleResponse = [&](auto sumQ, auto avgTOA, float k_fact) {
        for (auto j = 0; j < hgcaldigi::kTimeBins; j++) {
          sumQ(j, 0) = k_fact * sumQ(j, 0);
          avgTOA(j, 0) = k_fact * avgTOA(j, 0);

          auto en = sumQ(j, 0);
          if (en == 0) {
            avgTOA(j, 0) = 0.;
            continue;
          }

          avgTOA(j, 0) *= 1. / en;
        }
      };  //end scale response helper

      // dispatch operation
      for (auto idx : uniform_elements(acc, g4Hits.metadata().size())) {
        auto hit = g4Hits[idx];
        auto cellIdx = denseIndex[idx].cellInfoIdx();
        bool isSiPM = cellInfo[cellIdx].isSiPM();
        auto cce = conds[idx].signalScale();
        auto k_fact = cce / (isSiPM ? keV2MIP : keV2fC);

        scaleResponse(hit.sumQ(), hit.avgTOA(), k_fact);

      }  //end grid loop
    }
  };

  // @short apply all calibration kernels
  HGCalRealisticSimHitDevice HGCalAgeingAlgorithms::age(Queue& queue,
                                                        HGCalSimHitAccumHost const& host_g4Hits,
                                                        HGCalAgeingCondsDevice const& ageConds,
                                                        HGCalDenseIndexInfoDevice const& denseIndex,
                                                        HGCalMappingCellParamDevice const& cellInfo) const {
    edm::LogWarning("HGCalAgeingAlgorithms") << "Start of ageing: copying accumulators to device" << std::endl;
    auto const ncells = host_g4Hits.view().metadata().size();
    HGCalRealisticSimHitDevice device_g4Hits(queue, ncells);
    alpaka::memcpy(queue, device_g4Hits.buffer(), host_g4Hits.const_buffer());

    // number of items per group and division in groups to cover the whole problem
    uint32_t items = n_threads_;
    uint32_t groups = divide_up_by(ncells, items);

    // generate the working grid
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    edm::LogWarning("HGCalAgeingAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    //execute ageing kernels
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalAgeingKernel_finalizeResponse{},
                        device_g4Hits.view(),
                        ageConds.const_view(),
                        denseIndex.const_view(),
                        cellInfo.const_view());

    //FIXME - uncomment ifdef for final version
    //#ifdef EDM_ML_DEBUG
    int n_hits_to_print = 10;
    alpaka::exec<Acc1D>(queue, grid, HGCalAgeingKernel_printHits{}, device_g4Hits.view(), n_hits_to_print);
    //#endif

    return device_g4Hits;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
