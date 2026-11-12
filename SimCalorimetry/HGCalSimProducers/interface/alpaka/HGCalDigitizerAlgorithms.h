#ifndef SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalDigitizerAlgorithms_h
#define SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalDigitizerAlgorithms_h

// Alpaka imports
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// Host & device HGCal formats to handle
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalECONDPacketInfoDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalAgeingCondsDevice.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace hgcaldigi;
  using namespace hgcal;
  using namespace cms::alpakatools;

  /** 
    @short print realistic G4 hits for debugging purposes
  */
  struct HGCalAgeingKernel_printHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, HGCalRealisticSimHitDevice::ConstView view, int max_size) const {
      //lambdas for header and rows
      auto printHeader = [&](const char* label) {
        printf("| %s |", label);
        for (int i = 0; i < hgcaldigi::kTimeBins; i++) {
          printf(" %5d |", i);
        }
        printf("\n");
      };

      auto printRow = [&](const char* label, auto values) {
        printf("| %s |", label);
        for (int k = 0; k < hgcaldigi::kTimeBins; k++) {
          printf(" %.3f |", values(k, 0));
        }
        printf("\n");
      };

      //printout the required number of rows
      int n = 0;
      auto nhits = view.metadata().size();
      for (int i = 0; i < nhits; ++i) {
        auto const& hit = view[i];
        if (hit.sumQ().sum() == 0)
          continue;
        n++;
        if (n > max_size)
          break;
        printHeader("Bucket");
        printRow("q/p.e. * 1e3", 1e3 * hit.sumQ());
        printRow("<t> [ns]", hit.avgTOA());
      }
    }
  };

  class HGCalAgeingAlgorithms {
  public:
    HGCalAgeingAlgorithms(int n_blocks, int n_threads) : n_blocks_(n_blocks), n_threads_(n_threads) {}

    /**
      @short steer the applicaiton of ageing and other realistic reponse algorithms
    */
    HGCalRealisticSimHitDevice age(Queue&,
                                   HGCalSimHitAccumHost const&,
                                   HGCalAgeingCondsDevice const&,
                                   HGCalDenseIndexInfoDevice const&,
                                   HGCalMappingCellParamDevice const&) const;

  private:
    int n_blocks_, n_threads_;
  };

  class HGCalROCDigitizerAlgorithms {
  public:
    HGCalROCDigitizerAlgorithms(int n_blocks, int n_threads) : n_blocks_(n_blocks), n_threads_(n_threads) {}

    /**
      @short steer the digitization kernels for the emulation of the HGCROC
    */
    HGCalROCDigisAccumDevice digitize(Queue&,
                                      HGCalRealisticSimHitDevice&,
                                      HGCalECONDPacketInfoDevice const&,
                                      HGCalAgeingCondsDevice const&,
                                      HGCalDenseIndexInfoDevice const&,
                                      HGCalMappingCellParamDevice const&,
                                      const uint64_t,
                                      float const* crateJumps,
                                      uint16_t const* crateOfFed,
                                      uint32_t nFed,
                                      uint32_t nCrate) const;

    /**
      @short steer the harvesting of ROC digis for the in-time bunch
    */
    HGCalDigiDevice harvest(Queue&, HGCalROCDigisAccumDevice::ConstView) const;

  private:
    int n_blocks_, n_threads_;
  };

  class HGCalECONDAlgorithms {
  public:
    HGCalECONDAlgorithms(int n_blocks, int n_threads) : n_blocks_(n_blocks), n_threads_(n_threads) {}

    /**
      @short steer the filling of the ECONDPacketInfo
    */
    HGCalECONDPacketInfoDevice init(Queue&,
                                    HGCalAgeingCondsDevice const&,
                                    HGCalMappingModuleParamDevice const&,
                                    const uint64_t) const;

  private:
    int n_blocks_, n_threads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalDigitizerAlgorithms_h
