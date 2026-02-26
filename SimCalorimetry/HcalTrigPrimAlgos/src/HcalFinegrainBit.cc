#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>

std::bitset<2> HcalFinegrainBit::compute(const HcalFinegrainBit::Tower& tower) const {
  if (version_ == 0) {
    std::bitset<2> result;

    // First layer consistent with a MIP
    result[0] = tower[is_mip][0];

    // First layer consistent with a MIP, at least one layer with more
    // than MIP energy deposition
    result[1] = result[0] && (tower[is_above_mip].count() > 0);

    // There layers consistent with a MIP
    //    result[2] = tower[is_mip].count() >= 3;

    // Unset
    //    result[3] = false;

    return result;
  }
  if (version_ == 1) {
    std::bitset<2> result;

    // All algorithms the same for testing purposes
    result[0] = result[1] = tower[is_mip][0];

    return result;
  }
  if (version_ == 2) {
    std::bitset<2> result;

    // All algorithms the same for testing purposes
    result[0] = result[1] = true;

    return result;
  }
  return 0;
}

// timing/depth LLP bit setting
std::bitset<6> HcalFinegrainBit::compute(const HcalFinegrainBit::TowerTDC& tower, const HcalTrigTowerDetId& id) const {
  std::bitset<6> result;

  int tp_ieta = id.ieta();

  int Ndelayed = 0;
  int NveryDelayed = 0;
  int Nprompt = 0;
  int EarlyEnergy = 0;
  int DeepEnergy = 0;

  for (size_t i = 0; i < 7; i++) {
    int packedBits = tower[i].first.first;
    bool is_compressed = tower[i].first.second;
    int TDC = tower[i].second.first;
    int ADC = tower[i].second.second;

    // unpack packedBits
    int bit12_15set = (packedBits & 0xF);      // bits 0..3
    int hasTDCThr = (packedBits >> 20) & 0x1;  // bit 20
    int tdc1 = (packedBits >> 4) & 0xFF;       // bits 4..11
    int tdc2 = (packedBits >> 12) & 0xFF;      // bits 12..19

    int bit12 = (bit12_15set & 0b0001);       // low depth 1,2 energy
    int bit13 = (bit12_15set & 0b0010) >> 1;  // high depth 3+ energy
    int bit14 = (bit12_15set & 0b0100) >> 2;  // prompt energy passed
    int bit15 = (bit12_15set & 0b1000) >> 3;  // delayed energy passed

    // timing bits
    if (TDC < 50) {  // exclude error code for TDC in HE (unpacked)
      if ((abs(tp_ieta) <= 16) || (i >= 1)) {
        // count delayed / prompt hits either in HB, or in HE (excluding depth 1 due to backgrounds in HE)
        // Sim packing into Raw, has uncompressed TDC values (0-49) at the trigger primitive level. Packing (compressing HB TDC 6:2) happens in packer.
        // Hcal digis have compressed HB TDC (0-3)
        if (is_compressed == 1) {
          if (TDC == 1 && bit15 == 1)
            Ndelayed += 1;
          if (TDC == 2 && bit15 == 1)
            NveryDelayed += 1;
          if (TDC == 0 && bit14 == 1)
            Nprompt += 1;
        } else {
          // Read TDC thresholds from conditions. If not accessible, throw exception.
          if (!hasTDCThr) {
            throw cms::Exception("HBTDCThresholds")
                << "Missing or invalid TDC thresholds for uncompressed TP tower id=" << id << " and depthIndex=" << i
                << " (packedBits has hasTDCThr=0).";
          }

          if (TDC > tdc1 && TDC <= tdc2 && bit15 == 1)
            Ndelayed += 1;
          if (TDC > tdc2 && bit15 == 1)
            NveryDelayed += 1;
          if (TDC <= tdc1 && TDC >= 0 && bit14 == 1)
            Nprompt += 1;
        }
      }
    }

    // depth bit
    if (ADC > 0 && bit12 == 0)
      EarlyEnergy +=
          1;  // early layers, depth 1 and 2. If bit12 = 0, too much energy in early layers. Require ADC > 0 to ensure valid hit in cell
    if (ADC > 0 && bit13 == 1)
      DeepEnergy +=
          1;  // deep layers, 3+. If bit13 = 1, energy in deep layers. Require ADC > 0 to ensure valid hit in cell
  }

  // very delayed (001000), slightly delayed (000100), prompt (000010), depth flag (000001), 2 reserved bits (110000)
  if (DeepEnergy > 0 && EarlyEnergy == 0 && abs(tp_ieta) != 16)
    result[0] = true;  // 000001
  else
    result[0] = false;
  if (Nprompt > 0)
    result[1] = true;  // 000010
  else
    result[1] = false;
  if (Ndelayed > 0)
    result[2] = true;  // 000100
  else
    result[2] = false;
  if (NveryDelayed > 0)
    result[3] = true;  // 001000
  else
    result[3] = false;
  result[4] = result[5] = false;  // 110000 in HcalTriggerPrimitiveAlgo.cc, set to MIP bits from above

  return result;
}
