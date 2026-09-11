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
    //     0 --> number of photo-electrons (minus side),  1 --> time of flight (minus side)
    //     2 --> number of photo-electrons (plus side), 3 --> time of flight (plus side)
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
    BTLDigiContent()
        : rawId_(0),
          BC0count_(0),
          status_(false),
          BCcount_(0),
          chIDPlus_(0),
          T1coarsePlus_(0),
          T2coarsePlus_(0),
          EOIcoarsePlus_(0),
          ChargePlus_(0),
          T1finePlus_(0),
          T2finePlus_(0),
          IdleTimePlus_(0),
          PrevTrigFPlus_(0),
          TACIDPlus_(0),
          chIDMinus_(0),
          T1coarseMinus_(0),
          T2coarseMinus_(0),
          EOIcoarseMinus_(0),
          ChargeMinus_(0),
          T1fineMinus_(0),
          T2fineMinus_(0),
          IdleTimeMinus_(0),
          PrevTrigFMinus_(0),
          TACIDMinus_(0) {}

    uint32_t rawId_;
    uint16_t BC0count_;
    bool status_;
    uint32_t BCcount_;
    uint8_t chIDPlus_;       // TOFHIR channel ID, plus side of crystal
    uint16_t T1coarsePlus_;  // data from crystal plus side
    uint16_t T2coarsePlus_;
    uint16_t EOIcoarsePlus_;
    uint16_t ChargePlus_;
    uint16_t T1finePlus_;
    uint16_t T2finePlus_;
    uint16_t IdleTimePlus_;
    uint8_t PrevTrigFPlus_;
    uint8_t TACIDPlus_;
    uint8_t chIDMinus_;       // TOFHIR channel ID, minus side of crystal
    uint16_t T1coarseMinus_;  // data from crystal minus side
    uint16_t T2coarseMinus_;
    uint16_t EOIcoarseMinus_;
    uint16_t ChargeMinus_;
    uint16_t T1fineMinus_;
    uint16_t T2fineMinus_;
    uint16_t IdleTimeMinus_;
    uint8_t PrevTrigFMinus_;
    uint8_t TACIDMinus_;

    BTLDigiContent(uint32_t rawId,
                   uint16_t BC0count,
                   bool status,
                   uint32_t BCcount,
                   uint8_t chIDPlus,
                   uint16_t T1coarsePlus,
                   uint16_t T2coarsePlus,
                   uint16_t EOIcoarsePlus,
                   uint16_t ChargePlus,
                   uint16_t T1finePlus,
                   uint16_t T2finePlus,
                   uint16_t IdleTimePlus,
                   uint8_t PrevTrigFPlus,
                   uint8_t TACIDPlus,
                   uint8_t chIDMinus,
                   uint16_t T1coarseMinus,
                   uint16_t T2coarseMinus,
                   uint16_t EOIcoarseMinus,
                   uint16_t ChargeMinus,
                   uint16_t T1fineMinus,
                   uint16_t T2fineMinus,
                   uint16_t IdleTimeMinus,
                   uint8_t PrevTrigFMinus,
                   uint8_t TACIDMinus)
        : rawId_(rawId),
          BC0count_(BC0count),
          status_(status),
          BCcount_(BCcount),
          chIDPlus_(chIDPlus),
          T1coarsePlus_(T1coarsePlus),
          T2coarsePlus_(T2coarsePlus),
          EOIcoarsePlus_(EOIcoarsePlus),
          ChargePlus_(ChargePlus),
          T1finePlus_(T1finePlus),
          T2finePlus_(T2finePlus),
          IdleTimePlus_(IdleTimePlus),
          PrevTrigFPlus_(PrevTrigFPlus),
          TACIDPlus_(TACIDPlus),
          chIDMinus_(chIDMinus),
          T1coarseMinus_(T1coarseMinus),
          T2coarseMinus_(T2coarseMinus),
          EOIcoarseMinus_(EOIcoarseMinus),
          ChargeMinus_(ChargeMinus),
          T1fineMinus_(T1fineMinus),
          T2fineMinus_(T2fineMinus),
          IdleTimeMinus_(IdleTimeMinus),
          PrevTrigFMinus_(PrevTrigFMinus),
          TACIDMinus_(TACIDMinus) {}
  };

  struct ETLDigiContent {
    ETLDigiContent() : rawId_(0), header_(0), status_(0), colID_(0), rowID_(0), ToAdata_(0), ToTdata_(0), CALdata_(0) {}

    uint32_t rawId_;
    uint8_t header_;
    uint8_t status_;
    uint8_t colID_;
    uint8_t rowID_;
    uint16_t ToAdata_;
    uint16_t ToTdata_;
    uint16_t CALdata_;

    ETLDigiContent(uint32_t rawId,
                   uint8_t header,
                   uint8_t status,
                   uint8_t colID,
                   uint8_t rowID,
                   uint16_t ToAdata,
                   uint16_t ToTdata,
                   uint16_t CALdata)
        : rawId_(rawId),
          header_(header),
          status_(status),
          colID_(colID),
          rowID_(rowID),
          ToAdata_(ToAdata),
          ToTdata_(ToTdata),
          CALdata_(CALdata) {}
  };

  // use a wider integer now since we have to add row and column in an
  // intermediate det id for ETL
  typedef std::unordered_map<MTDCellId, MTDCellInfo> MTDSimHitDataAccumulator;
  typedef std::vector<BTLDigiContent> BTLDigiTempCollection;  // temporary collection to store BTL digis in SoA format
  typedef std::vector<ETLDigiContent> ETLDigiTempCollection;  // temporary collection to store ETL digis in SoA format

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
