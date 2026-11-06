#include <cstdint>
#include <iostream>
#include <cassert>
#include <string>
#include <random>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "SimCalorimetry/HGCalSimAlgos/interface/alpaka/HGCROCChannelFramer.h"

namespace {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi;
}

TEST_CASE("SimCalorimetry/HGCalSimAlgos test", "[HGCROCChannelFramer]") {
  SECTION("Test HGCROCChannelFramer round trip") {
    REQUIRE_NOTHROW([&]() {
      // http://www.cplusplus.com/reference/random/linear_congruential_engine/
      unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
      std::minstd_rand0 myrand(seed1);

      //wrap the procedure to assert TOT read is within quantization error
      //a truncation of two bits leads to an uncertainty of +/- 4 ADC counts
      auto totOK = [&](uint16_t tot_orig, uint16_t tot_read) {
        int delta(tot_read - tot_orig);
        return (delta >= -4 && delta <= 4);
      };

      // do the trials: time/performance test and exploit randomisation to check
      constexpr uint32_t ntrials = 1000000;
      for (uint32_t i = 0; i < ntrials; i++) {
        //a randomiset set of values
        uint16_t adc = myrand() % 1024;
        uint16_t adcm1 = myrand() % 1024;
        uint16_t tot = myrand() % 2048;
        uint16_t toa = myrand() % 1024;
        bool tc = myrand() % 2;
        bool tp = myrand() % 2;
        uint16_t tctp = (tp | (tc << 1));
        bool charMode = myrand() % 2;

        //build the ROC word
        const uint32_t packed = rocFrame(tc, tp, adcm1, adc, tot, toa, charMode);

        //check if the unpacking matches the original generated values
        if (charMode) {
          REQUIRE(unpackADC_charMode(packed) == adc);
          REQUIRE(unpackADCm1_charMode(packed) == 0);
          REQUIRE(unpackTOA(packed) == toa);
          REQUIRE(unpackTcTp(packed) == tctp);
          REQUIRE(totOK(unpackTOT_charMode(packed), tot));
        } else {
          if (tc) {
            REQUIRE(unpackADC(packed) == 0);
            REQUIRE(unpackADCm1(packed) == adcm1);
            REQUIRE(unpackTOA(packed) == toa);
            REQUIRE(unpackTcTp(packed) == tctp);
            REQUIRE(totOK(unpackTOT(packed), tot));
          } else {
            REQUIRE(unpackADC(packed) == adc);
            REQUIRE(unpackADCm1(packed) == adcm1);
            REQUIRE(unpackTOA(packed) == toa);
            REQUIRE(unpackTcTp(packed) == tctp);
            REQUIRE(totOK(unpackTOT(packed), 0));
          }
        }

      }  // end of trials
    }());
  }  // end SECTION
}  // end TEST_CASE
