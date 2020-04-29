#ifndef __SimCalorimetry_HGCCalSimProducers_HGCDigitizerTypes_h__
#define __SimCalorimetry_HGCCalSimProducers_HGCDigitizerTypes_h__

#include <unordered_map>
#include <array>
#include <functional>

#include "DataFormats/DetId/interface/DetId.h"

namespace hgc_digi {

  //15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
  constexpr size_t nSamples = 15;
  constexpr size_t npreHits = 30;
  typedef float HGCSimData_t;

  typedef std::array<HGCSimData_t, nSamples> HGCSimHitData;
  //typedef std::array<HGCSimData_t, npreHits> PUSimHitData;// to keep 30 hits while incoming hits shape up to signal
  //typedef std::array<PUSimHitData, nSamples> PUSimHitContainer;// Container of RawSimHitData

  typedef std::vector<HGCSimData_t> vec;
  typedef std::array<vec, nSamples> PUSimHitData;

  typedef std::vector<std::pair<float, float> > hitsRecordData;
  typedef std::array<hitsRecordData, nSamples> hitsRecordForMultipleBxs;

  struct HGCCellHitInfo {
    std::array<PUSimHitData, 2> PUhit_info;
    int thickness;
    double size;
    hitsRecordForMultipleBxs hitsRecord;
  };
  struct HGCCellInfo {
    //1st array=energy, 2nd array=energy weighted time-of-flight
    std::array<HGCSimHitData, 2> hit_info;
    int thickness;
    double size;
  };

  typedef std::unordered_map<uint32_t, HGCCellInfo> HGCSimHitDataAccumulator;
  typedef std::unordered_map<uint32_t, HGCCellHitInfo> HGCPUSimHitDataAccumulator;
}  // namespace hgc_digi
#endif
