#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include <cassert>

std::bitset<2> HcalFinegrainBit::compute(const HcalFinegrainBit::Tower& tower) const {
  if (version_ == 0) {
    std::bitset<2> result;

    // First layer consistent with a MIP
    result[0] = tower[is_mip][0];

    // First layer consistent with a MIP, at least one layer with more
    // than MIP energy deposition
    result[1] = result[0] & (tower[is_above_mip].count() > 0);

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
  const int MinE = 64;

  int DeepEnergy = 0;
  int EarlyEnergy = 0;
  const int deepLayerMinE = 80;
  const int earlyLayerMaxE = 16;

  for (size_t i = 0; i < 7; i++) {
    int ADC = tower[i].first;
    int TDC = tower[i].second;

    // timing bits
    if (TDC < 50) {  // exclude error code for TDC in HE (unpacked)
      if (abs(tp_ieta) <=
          16) {  // in HB, TDC values are compressed. 01 = first delayed range, 10 = second delayed range
        if (TDC == 1 && ADC >= MinE)
          Ndelayed += 1;
        if (TDC == 2 && ADC >= MinE)
          NveryDelayed += 1;
        if (TDC == 0 && ADC >= MinE)
          Nprompt += 1;
      }
      if (abs(tp_ieta) > 16 &&
          i >= 1) {  // in HE, TDC values are uncompressed (0-49). Exclude depth 1 in HE due to backgrounds
        if (TDC > tdc_HE[abs(tp_ieta) - 1][i] && TDC <= tdc_HE[abs(tp_ieta) - 1][i] + 2 && ADC >= MinE)
          Ndelayed += 1;
        if (TDC > tdc_HE[abs(tp_ieta) - 1][i] + 2 && ADC >= MinE)
          NveryDelayed += 1;
        if (TDC <= tdc_HE[abs(tp_ieta) - 1][i] && TDC >= 0 && ADC >= MinE)
          Nprompt += 1;
      }
    }

    // depth bit
    if (i <= 1 && ADC >= earlyLayerMaxE)
      EarlyEnergy += 1;  // early layers, depth 1 and 2
    if (i >= 2 && ADC >= deepLayerMinE)
      DeepEnergy += 1;  // deep layers, 3+
  }

  // very delayed (100000), slightly delayed (010000), prompt (001000), 2 reserved bits (000110), depth flag (000001)
  if (DeepEnergy > 0 && EarlyEnergy == 0)
    result[0] = true;  // 000001
  else
    result[0] = false;
  if (Nprompt > 0)
    result[3] = true;  // 001000
  else
    result[3] = false;
  if (Ndelayed > 0)
    result[4] = true;  // 010000
  else
    result[4] = false;
  if (NveryDelayed > 0)
    result[5] = true;  // 100000
  else
    result[5] = false;
  result[1] = result[2] = false;  // 000110 in HcalTriggerPrimitiveAlgo.cc, set to MIP bits from above

  return result;
}
