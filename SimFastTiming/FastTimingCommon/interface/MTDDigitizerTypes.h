#ifndef __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__
#define __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__

#include <unordered_map>
#include <array>

namespace mtd_digitizer {

  //15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
  constexpr size_t nSamples = 15;
  
  typedef float MTDSimData_t;
  
  typedef std::array<MTDSimData_t,nSamples> MTDSimHitData;  
  
  struct MTDCellInfo {
    //1st array=energy, 2nd array=time-of-flight
    std::array<MTDSimHitData,2> hit_info;    
  };
  
  typedef std::unordered_map<uint32_t, MTDCellInfo > MTDSimHitDataAccumulator; 

}
#endif
