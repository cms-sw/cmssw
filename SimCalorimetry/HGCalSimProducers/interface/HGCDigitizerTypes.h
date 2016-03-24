#ifndef __SimCalorimetry_HGCCalSimProducers_HGCDigitizerTypes_h__
#define __SimCalorimetry_HGCCalSimProducers_HGCDigitizerTypes_h__

#include <unordered_map>
#include <array>

namespace hgc_digi {

  //15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
  constexpr size_t nSamples = 15;
  
  typedef float HGCSimData_t;
  
  typedef std::array<HGCSimData_t,nSamples> HGCSimHitData;  
  
  struct HGCCellInfo {
    //1st array=energy, 2nd array=energy weighted time-of-flight
    std::array<HGCSimHitData,2> hit_info;
    int thickness;
  };
  
  typedef std::unordered_map<uint32_t, HGCCellInfo > HGCSimHitDataAccumulator; 

}
#endif
