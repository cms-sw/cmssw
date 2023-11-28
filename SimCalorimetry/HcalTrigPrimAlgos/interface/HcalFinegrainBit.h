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

  HcalFinegrainBit(int version) : version_(version){};

  std::bitset<2> compute(const Tower&) const;
  std::bitset<6> compute(const TowerTDC&, const HcalTrigTowerDetId&) const;

private:
  // define the two bits in the tower
  const int is_mip = 0;
  const int is_above_mip = 1;

  int version_;

  // define prompt-delayed TDC range. Note this is offset from depth and ieta by 1
  const int tdc_boundary[29][7] = {
      {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0},
      {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0},
      {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0},
      {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0}, {12, 12, 12, 12, 0, 0, 0},
      {0, 12, 10, 0, 0, 0, 0},   {0, 9, 10, 9, 10, 0, 0},   {16, 9, 9, 9, 11, 10, 0},  {17, 8, 9, 8, 9, 10, 0},
      {9, 7, 7, 7, 9, 6, 0},     {8, 7, 7, 6, 6, 6, 0},     {8, 6, 6, 6, 7, 7, 0},     {7, 6, 6, 6, 7, 6, 0},
      {7, 6, 6, 6, 6, 6, 0},     {6, 6, 6, 6, 6, 6, 0},     {6, 5, 6, 6, 6, 7, 10},    {9, 9, 9, 5, 5, 6, 6},
      {0, 0, 0, 0, 0, 0, 0}};
};

#endif
