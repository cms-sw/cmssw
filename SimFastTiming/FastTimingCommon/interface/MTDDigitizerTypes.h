#ifndef __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__
#define __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__

#include "DataFormats/DetId/interface/DetId.h"
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

  struct MTDCellId {
    MTDCellId() : detid_(0), row_(0), column_(0) {}
    const uint32_t detid_;
    const uint8_t row_, column_;
    MTDCellId(const DetId& id) : detid_(id.rawId()), row_(0), column_(0) {}
    MTDCellId(const DetId& id, uint8_t row, uint8_t col) : detid_(id.rawId()), row_(row), column_(col) {} 
    bool operator==(const MTDCellId& eq) const {
      return (detid_ == eq.detid_) && (row_ == eq.row_) && (column_ == eq.column_);
    }
  };
  
  // use a wider integer now since we have to add row and column in an
  // intermediate det id for ETL
  typedef std::unordered_map<MTDCellId, MTDCellInfo> MTDSimHitDataAccumulator; 

}

namespace std
{

  constexpr int kRowOffset = 32;
  constexpr int kColOffset = 40;

  template<> struct hash<mtd_digitizer::MTDCellId>
    {
      typedef mtd_digitizer::MTDCellId argument_type;
      typedef std::size_t result_type;
      result_type operator()(argument_type const& s) const noexcept
      {
	uint64_t input = (uint64_t)s.detid_ | ((uint64_t)s.row_) << kRowOffset | ((uint64_t)s.column_) << kColOffset;
	return std::hash<uint64_t>()(input); 
      }
    };
}

#endif
