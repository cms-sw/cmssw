#ifndef __SimCalorimetry_FastTimingSimProducers_FTLDigitizerTypes_h__
#define __SimCalorimetry_FastTimingSimProducers_FTLDigitizerTypes_h__

#include <unordered_map>
#include <array>

namespace ftl_digitizer {

  //15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
  constexpr size_t nSamples = 15;

  typedef float FTLSimData_t;

  typedef std::array<FTLSimData_t, nSamples> FTLSimHitData;

  struct FTLCellInfo {
    //1st array=energy, 2nd array=energy weighted time-of-flight
    std::array<FTLSimHitData, 2> hit_info;
  };

  typedef std::unordered_map<uint32_t, FTLCellInfo> FTLSimHitDataAccumulator;

}  // namespace ftl_digitizer
#endif
