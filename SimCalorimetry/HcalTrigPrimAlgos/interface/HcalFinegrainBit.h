#ifndef HcalSimAlgos_HcalFinegrainBit_h
#define HcalSimAlgos_HcalFinegrainBit_h

#include <array>
#include <bitset>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

class HcalFinegrainBit {
public:
  // see the const definitions below for the meaning of the bit towers.
  // Each bit is replicated for each depth level
  typedef std::array<std::bitset<6>, 2> Tower;
  // Each pair contains uHTR group 0 LUT bits 12-15, TDC, and ADC of the cell in that depth of the trigger tower
  typedef std::array<std::pair<std::pair<int, bool>, std::pair<int, int>>, 7> TowerTDC;

  HcalFinegrainBit(int version) : version_(version) {}

  std::bitset<2> compute(const Tower&) const;
  std::bitset<6> compute(const TowerTDC&, const HcalTrigTowerDetId&) const;

private:
  // define the two bits in the tower
  const int is_mip = 0;
  const int is_above_mip = 1;

  int version_;
};

#endif
