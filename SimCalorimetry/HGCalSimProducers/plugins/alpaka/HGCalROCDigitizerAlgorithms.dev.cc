#include <cstddef>
#include <cstdio>
#include <sstream>
#include <cmath>

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// HGCal imports
#include "SimCalorimetry/HGCalSimProducers/interface/alpaka/HGCalDigitizerAlgorithms.h"
#include "SimCalorimetry/HGCalSimAlgos/interface/alpaka/HGCROCChannelFramer.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // @short this kernel takes care of adding noise on each bunch crossing
  struct HGCalROCDigitizerKernel_addNoise {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalRealisticSimHitDevice::View realisticHits,
                                  HGCalECONDPacketInfoDevice::ConstView econds,
                                  HGCalAgeingCondsDevice::ConstView conds,
                                  HGCalDenseIndexInfoDevice::ConstView denseIndex,
                                  HGCalMappingCellParamDevice::ConstView cellInfo,
                                  const uint64_t seed) const {
      const uint32_t globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      // one engine per thread (or per work item)
      auto engine = alpaka::rand::engine::createDefault(acc, seed + globalThreadIdx, 0);
      auto normal = alpaka::rand::distribution::createNormalReal<float>(acc);

      // dispatch operation
      for (auto idx : uniform_elements(acc, realisticHits.metadata().size())) {
        auto sumQ = realisticHits[idx].sumQ();
        auto enc = conds[idx].enc();
        //auto cellIdx = denseIndex[idx].cellInfoIdx();
        //bool isSiPM = cellInfo[cellIdx].isSiPM();

        //loop over time slices
        for (auto j = 0; j < hgcaldigi::kTimeBins; j++) {
          const float r = normal(engine);
          sumQ(j, 0) += r * enc;
        }  //end time slices loop

      }  //end grid loop
    }
  };

  // @short add the phase shift of this cell due to "BE crate jumps" + LHC shift
  struct HGCalROCDigitizerKernel_emulatePhaseShits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalRealisticSimHitDevice::View realisticHits,
                                  HGCalDenseIndexInfoDevice::ConstView denseIndex,
                                  HGCalPhaseShiftsDevice::ConstView phaseShifts,
                                  float lhc_phaseshift) const {
      for (auto idx : uniform_elements(acc, realisticHits.metadata().size())) {
        const float deltaTOA = phaseShifts[idx].deltaT();
        auto toa = realisticHits[idx].avgTOA();
        auto fed = denseIndex[idx].fedId();

        for (auto j = 0; j < hgcaldigi::kTimeBins; j++) {
          const float t = toa(j, 0);
          if (t != 0.f) {
            toa(j, 0) = t + deltaTOA + lhc_phaseshift;
#ifdef EDM_ML_DEBUG
            printf("cell %u fed %u avgTOA = %f -> %f [ns]\n", static_cast<unsigned>(idx), fed, t, toa(j, 0));
#endif
          }
        }
      }
    }
  };

  // @short this kernel takes care of finalizing the Geant4 energy response and assigning the final energy deposit and average time estimates
  struct HGCalROCDigitizerKernel_digitize {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalROCDigisAccumDevice::View rocDigis,
                                  HGCalRealisticSimHitDevice::ConstView realisticHits,
                                  HGCalAgeingCondsDevice::ConstView conds,
                                  HGCalDenseIndexInfoDevice::ConstView denseIndex,
                                  HGCalMappingCellParamDevice::ConstView cellInfo) const {
      auto quantize = [](float value, float lsb, int maxCode) -> uint32_t {
        if (!(lsb > 0.f))
          return 0u;
        int code = static_cast<int>(std::round(value / lsb));
        if (!(code > 0.f))
          return 0u;
        return static_cast<uint32_t>(code > maxCode ? maxCode : code);
      };

      // build the energy part of the word
      auto digitize = [&](auto word,
                          const auto sumQ,
                          const auto avgTOA,
                          const float& adc_lsb,
                          const float& tot_lsb,
                          const float& tot_thr,
                          const float& toa_thr) {
        constexpr float toa_lsb = 25. / 1024;

        for (auto j = 0; j < hgcaldigi::kTimeBins; j++) {
          auto q = sumQ(j, 0);
          auto qm1 = j > 0 ? sumQ(j - 1, 0) : 0;
          auto t = avgTOA(j, 0);
          uint16_t adc = static_cast<uint16_t>(quantize(q, adc_lsb, 1023));
          uint16_t tot = static_cast<uint16_t>(quantize(q, tot_lsb, 4095));
          uint16_t adcm1 = static_cast<uint16_t>(j > 0 ? quantize(qm1, adc_lsb, 1023) : 0);
          bool tc = (q > tot_thr);
          bool tp = tc;
          uint16_t toa = static_cast<uint16_t>(q > toa_thr ? quantize(t, toa_lsb, 1023) : 0);

          //compress all to 32b
          word(j, 0) = hgcaldigi::rocFrame(tc, tp, adcm1, adc, tot, toa, false);
        }
      };

      // dispatch operation
      for (auto idx : uniform_elements(acc, realisticHits.metadata().size())) {
        auto digi = rocDigis[idx];
        const auto hit = realisticHits[idx];
        digitize(digi.word(),
                 hit.sumQ(),
                 hit.avgTOA(),
                 conds[idx].adc_lsb(),
                 conds[idx].tot_lsb(),
                 conds[idx].tot_thr(),
                 conds[idx].toa_thr());
      }  //end grid loop
    }
  };

  // @short this kernel takes care of finalizing the Geant4 energy response and assigning the final energy deposit and average time estimates
  struct HGCalROCDigitizerKernel_harvestROCDigis {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalDigiDevice::View rocDigis,
                                  HGCalROCDigisAccumDevice::ConstView acc_rocDigis,
                                  int index) const {
      // dispatch operation
      for (auto idx : uniform_elements(acc, acc_rocDigis.metadata().size())) {
        auto acc_digi = acc_rocDigis[idx];
        const uint32_t word = acc_digi.word()(index, 0);

        auto digi = rocDigis[idx];
        digi.tctp() = hgcaldigi::unpackTcTp(word);
        digi.adcm1() = hgcaldigi::unpackADCm1(word);
        digi.adc() = hgcaldigi::unpackADC(word);
        digi.tot() = hgcaldigi::unpackTOT(word);
        digi.toa() = hgcaldigi::unpackTOA(word);
        digi.cm() = 0;
        digi.flags() = hgcal::DIGI_FLAG::FULL_READOUT;
      }  //end grid loop
    }
  };

  // @short apply the sequence of digitization kernels
  HGCalROCDigisAccumDevice HGCalROCDigitizerAlgorithms::digitize(Queue& queue,
                                                                 HGCalRealisticSimHitDevice& device_realisticHits,
                                                                 HGCalECONDPacketInfoDevice const& device_econds,
                                                                 HGCalAgeingCondsDevice const& ageConds,
                                                                 HGCalDenseIndexInfoDevice const& denseIndex,
                                                                 HGCalMappingCellParamDevice const& cellInfo,
                                                                 HGCalPhaseShiftsDevice const& phaseShifts,
                                                                 double lhc_phaseshift,
                                                                 const uint64_t seed) const {
    LogDebug("HGCalROCDigitizerAlgorithms") << "Start of digitization" << std::endl;
    auto const ncells = device_realisticHits.view().metadata().size();
    HGCalROCDigisAccumDevice device_rocDigis(queue, ncells);

    // number of items per group and division in groups to cover the whole problem
    uint32_t items = n_threads_;
    uint32_t groups = divide_up_by(ncells, items);

    // generate the working grid
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalROCDigitizerAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    //execute digitization kernels
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalROCDigitizerKernel_addNoise{},
                        device_realisticHits.view(),
                        device_econds.const_view(),
                        ageConds.const_view(),
                        denseIndex.const_view(),
                        cellInfo.const_view(),
                        seed);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalROCDigitizerKernel_emulatePhaseShits{},
                        device_realisticHits.view(),
                        denseIndex.const_view(),
                        phaseShifts.const_view(),
                        lhc_phaseshift);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalROCDigitizerKernel_digitize{},
                        device_rocDigis.view(),
                        device_realisticHits.const_view(),
                        ageConds.const_view(),
                        denseIndex.const_view(),
                        cellInfo.const_view());

#ifdef EDM_ML_DEBUG
    int n_hits_to_print = 10;
    alpaka::exec<Acc1D>(queue, grid, HGCalAgeingKernel_printHits{}, device_realisticHits.view(), n_hits_to_print);
#endif

    return device_rocDigis;
  }

  // @short apply all calibration kernels
  HGCalDigiDevice HGCalROCDigitizerAlgorithms::harvest(Queue& queue,
                                                       HGCalROCDigisAccumDevice::ConstView device_acc_rocDigis) const {
    LogDebug("HGCalROCDigitizerAlgorithms") << "Start of harvest" << std::endl;
    auto const ncells = device_acc_rocDigis.metadata().size();
    HGCalDigiDevice device_rocDigis(queue, ncells);

    // number of items per group and division in groups to cover the whole problem
    uint32_t items = n_threads_;
    uint32_t groups = divide_up_by(ncells, items);

    // generate the working grid
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalROCDigitizerAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    //execute digitization kernels
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalROCDigitizerKernel_harvestROCDigis{},
                        device_rocDigis.view(),
                        device_acc_rocDigis,
                        ::hgcaldigi::kInTimeBin);

    return device_rocDigis;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
