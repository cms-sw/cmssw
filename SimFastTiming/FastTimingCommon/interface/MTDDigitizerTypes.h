#ifndef __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__
#define __SimCalorimetry_FastTimingSimProducers_MTDDigitizerTypes_h__

#include "DataFormats/DetId/interface/DetId.h"
#include <unordered_map>
#include <array>

namespace mtd_digitizer {

  //15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
  constexpr size_t nSamples = 15;

  typedef float MTDSimData_t;

  typedef std::array<MTDSimData_t, nSamples> MTDSimHitData;

  struct MTDCellInfo {
    // for BTL:
    //     0 --> number of photo-electrons (left side),  1 --> time of flight (left side)
    //     2 --> number of photo-electrons (right side), 3 --> time of flight (right side)
    std::array<MTDSimHitData, 4> hit_info;
  };

  // Maximum value of time of flight for premixing packing
  constexpr float PREMIX_MAX_TOF = 25.0f;

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

  struct BTLDigiContent {
    
    BTLDigiContent() : rawId_(0), BC0count_(0), status_(false), BCcount_(0), chIDR_(0), T1coarseR_(0), T2coarseR_(0), EOIcoarseR_(0), ChargeR_(0), T1fineR_(0), T2fineR_(0), IdleTimeR_(0), PrevTrigFR_(0), TACIDR_(0), chIDL_(0), T1coarseL_(0), T2coarseL_(0), EOIcoarseL_(0), ChargeL_(0), T1fineL_(0), T2fineL_(0), IdleTimeL_(0), PrevTrigFL_(0), TACIDL_(0) {}
    
    uint32_t rawId_;
    uint16_t BC0count_;
    bool status_;
    uint32_t BCcount_;
    uint8_t chIDR_;       // TOFHIR channel ID, right side of crystal
    uint16_t T1coarseR_;  // data from crystal right side
    uint16_t T2coarseR_;
    uint16_t EOIcoarseR_;
    uint16_t ChargeR_;
    uint16_t T1fineR_;
    uint16_t T2fineR_;
    uint16_t IdleTimeR_;
    uint8_t PrevTrigFR_;
    uint8_t TACIDR_;
    uint8_t chIDL_;       // TOFHIR channel ID, left side of crystal
    uint16_t T1coarseL_;  // data from crystal left side
    uint16_t T2coarseL_;
    uint16_t EOIcoarseL_;
    uint16_t ChargeL_;
    uint16_t T1fineL_;
    uint16_t T2fineL_;
    uint16_t IdleTimeL_;
    uint8_t PrevTrigFL_;
    uint8_t TACIDL_;

    BTLDigiContent(uint32_t rawId, uint16_t BC0count, bool status, uint32_t BCcount, uint8_t chIDR, uint16_t T1coarseR, uint16_t T2coarseR, uint16_t EOIcoarseR, uint16_t ChargeR, uint16_t T1fineR, uint16_t T2fineR, uint16_t IdleTimeR, uint8_t PrevTrigFR, uint8_t TACIDR, uint8_t chIDL, uint16_t T1coarseL, uint16_t T2coarseL, uint16_t EOIcoarseL, uint16_t ChargeL, uint16_t T1fineL, uint16_t T2fineL, uint16_t IdleTimeL, uint8_t PrevTrigFL, uint8_t TACIDL) :
      rawId_(rawId), BC0count_(BC0count), status_(status), BCcount_(BCcount), chIDR_(chIDR), T1coarseR_(T1coarseR), T2coarseR_(T2coarseR), EOIcoarseR_(EOIcoarseR), ChargeR_(ChargeR), T1fineR_(T1fineR), T2fineR_(T2fineR), IdleTimeR_(IdleTimeR), PrevTrigFR_(PrevTrigFR), TACIDR_(TACIDR), chIDL_(chIDL), T1coarseL_(T1coarseL), T2coarseL_(T2coarseL), EOIcoarseL_(EOIcoarseL), ChargeL_(ChargeL), T1fineL_(T1fineL), T2fineL_(T2fineL), IdleTimeL_(IdleTimeL), PrevTrigFL_(PrevTrigFL), TACIDL_(TACIDL) {} 
  };


  // use a wider integer now since we have to add row and column in an
  // intermediate det id for ETL
  typedef std::unordered_map<MTDCellId, MTDCellInfo> MTDSimHitDataAccumulator;
  typedef std::vector<BTLDigiContent> BTLDigiTempCollection; // temporary collection to store BTL digis in SoA format

  constexpr int kNumberOfBX = 15;
  constexpr int kInTimeBX = 9;

}  // namespace mtd_digitizer

namespace std {

  constexpr int kRowOffset = 32;
  constexpr int kColOffset = 40;

  template <>
  struct hash<mtd_digitizer::MTDCellId> {
    typedef mtd_digitizer::MTDCellId argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& s) const noexcept {
      uint64_t input = (uint64_t)s.detid_ | ((uint64_t)s.row_) << kRowOffset | ((uint64_t)s.column_) << kColOffset;
      return std::hash<uint64_t>()(input);
    }
  };
}  // namespace std

#endif
