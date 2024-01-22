// Author: Aurora Perego, Fabio Cossutti - aurora.perego@cern.ch, fabio.cossutti@ts.infn.it
// Date: 05/2023

#ifndef SimDataFormats_CaloAnalysis_MtdSimCluster_h
#define SimDataFormats_CaloAnalysis_MtdSimCluster_h

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <vector>

class MtdSimCluster : public SimCluster {
  friend std::ostream &operator<<(std::ostream &s, MtdSimCluster const &tp);

public:
  MtdSimCluster();
  MtdSimCluster(const SimTrack &simtrk);
  MtdSimCluster(EncodedEventId eventID, uint32_t particleID);  // for PU

  // destructor
  ~MtdSimCluster();

  /** @brief add hit time */
  void addHitTime(float time) {
    times_.emplace_back(time);
    ++nsimhits_;
  }

  /** @brief add hit with fraction */
  void addHitAndFraction(uint64_t hit, float fraction) {
    mtdHits_.emplace_back(hit);
    fractions_.emplace_back(fraction);
  }

  /** @brief Returns list of hit IDs and fractions for this SimCluster */
  std::vector<std::pair<uint64_t, float>> hits_and_fractions() const {
    assert(mtdHits_.size() == fractions_.size());
    std::vector<std::pair<uint64_t, float>> result;
    result.reserve(mtdHits_.size());
    for (size_t i = 0; i < mtdHits_.size(); ++i) {
      result.emplace_back(mtdHits_[i], fractions_[i]);
    }
    return result;
  }

  /** @brief Returns list of hit IDs and energies for this SimCluster */
  std::vector<std::pair<uint64_t, float>> hits_and_energies() const {
    assert(mtdHits_.size() == energies_.size());
    std::vector<std::pair<uint64_t, float>> result;
    result.reserve(mtdHits_.size());
    for (size_t i = 0; i < mtdHits_.size(); ++i) {
      result.emplace_back(mtdHits_[i], energies_[i]);
    }
    return result;
  }

  /** @brief clear the hits and fractions list */
  void clearHitsAndFractions() {
    std::vector<uint64_t>().swap(mtdHits_);
    std::vector<float>().swap(fractions_);
  }

  /** @brief Returns list of hit IDs and times for this SimCluster */
  std::vector<std::pair<uint64_t, float>> hits_and_times() const {
    assert(mtdHits_.size() == times_.size());
    std::vector<std::pair<uint64_t, float>> result;
    result.reserve(mtdHits_.size());
    for (size_t i = 0; i < mtdHits_.size(); ++i) {
      result.emplace_back(mtdHits_[i], times_[i]);
    }
    return result;
  }

  /** @brief Returns list of detIds, rows and columns for this SimCluster */
  std::vector<std::pair<uint32_t, std::pair<uint8_t, uint8_t>>> detIds_and_rows() const {
    std::vector<std::pair<uint32_t, std::pair<uint8_t, uint8_t>>> result;
    result.reserve(mtdHits_.size());
    for (size_t i = 0; i < mtdHits_.size(); ++i) {
      result.emplace_back(
          mtdHits_[i] >> 32,
          std::pair<uint8_t, uint8_t>(static_cast<uint8_t>(mtdHits_[i] >> 16), static_cast<uint8_t>(mtdHits_[i])));
    }
    return result;
  }

  /** @brief clear the times list */
  void clearHitsTime() { std::vector<float>().swap(times_); }

  void clear() {
    clearHitsAndFractions();
    clearHitsEnergy();
    clearHitsTime();
  }

  /** @brief add simhit's energy to cluster */
  void addSimHit(const PSimHit &hit) {
    simhit_energy_ += hit.energyLoss();
    ++nsimhits_;
  }

  // --  for backward compatibility
  void newForm() {
    static const uint32_t kMTDsubdOffset = 23;
    static const uint32_t kMTDsubdMask = 0x3;

    static constexpr uint32_t kBTLmoduleOffset = 10;
    static constexpr uint32_t kBTLmoduleMask = 0x3F;
    static constexpr uint32_t kBTLmodTypeOffset = 8;
    static constexpr uint32_t kBTLmodTypeMask = 0x3;
    static constexpr uint32_t kModulePerTypeBarPhiFlat = 48 / 3;

    static constexpr uint32_t kBTLRUOffset = 6;
    static constexpr uint32_t kBTLRUMask = 0x3;
    static constexpr uint32_t kBTLCrystalMask = 0x3F;
    static constexpr uint32_t kCrystalsPerModuleV2 = 16;

    for (size_t i = 0; i < mtdHits_.size(); ++i) {
      // - Get detId, row, column from old format
      uint32_t rawid = mtdHits_[i] >> 32;
      uint8_t row = static_cast<uint8_t>(mtdHits_[i] >> 16);
      uint8_t col = static_cast<uint8_t>(mtdHits_[i]);

      uint32_t newid = 0;

      // - Change bit content only for BTL ( MTD subdetector BTL/ETL : bit 24-23)
      if (((rawid >> kMTDsubdOffset) & kMTDsubdMask) == 1) {
        // - check bit 6-7: if = 0 --> old barphiflat BTL layout, if !=0 V2, V3 BTL layout
        if (((rawid >> kBTLRUOffset) & kBTLRUMask) == 0) {
          // - change bit fields for module (15-10), crystal type (9-8), crystal num (7-0) as done in BTLDetId.cc for geom V1
          uint32_t module = (rawid >> kBTLmoduleOffset) & kBTLmoduleMask;
          uint32_t modtype = (rawid >> kBTLmodTypeOffset) & kBTLmodTypeMask;
          uint32_t newmodule = module + kModulePerTypeBarPhiFlat * (modtype - 1);
          newid = ((rawid >> kBTLmoduleOffset) & ~kBTLmoduleMask) | newmodule;
          newid = newid << kBTLmoduleOffset;
        } else {
          // - change bit field for crystal number to 17-1 for geom V2, V3
          newid = (~kBTLCrystalMask & rawid) | kCrystalsPerModuleV2;
        }

        // - Create the unique id in the new format
        uint64_t uniqueIdNew = static_cast<uint64_t>(newid) << 32;
        uniqueIdNew |= row << 16;
        uniqueIdNew |= col;

        mtdHits_[i] = uniqueIdNew;
      }
    }
  }

protected:
  std::vector<uint64_t> mtdHits_;
  std::vector<float> times_;
};

#endif
