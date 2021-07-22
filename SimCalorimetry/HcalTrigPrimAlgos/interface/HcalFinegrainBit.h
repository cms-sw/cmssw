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
  // Each pair contains energy and TDC of the cell in that depth of the trigger tower
  typedef std::array<std::pair<int, int>, 7> TowerTDC;

  HcalFinegrainBit(int version) : version_(version){};

  std::bitset<2> compute(const Tower&) const;
  std::bitset<6> compute(const TowerTDC&, const HcalTrigTowerDetId&) const;

private:
  // define the two bits in the tower
  const int is_mip = 0;
  const int is_above_mip = 1;

  int version_;

  // define prompt-delayed TDC range. Note this is offset from depth and ieta by 1
  const int tdc_HE[29][7] = {
      {8, 14, 15, 17, 0, 0, 0}, {8, 14, 15, 17, 0, 0, 0}, {8, 14, 14, 17, 0, 0, 0}, {8, 14, 14, 17, 0, 0, 0},
      {8, 13, 14, 16, 0, 0, 0}, {8, 13, 14, 16, 0, 0, 0}, {8, 12, 14, 15, 0, 0, 0}, {8, 12, 14, 15, 0, 0, 0},
      {7, 12, 13, 15, 0, 0, 0}, {7, 12, 13, 15, 0, 0, 0}, {7, 12, 13, 15, 0, 0, 0}, {7, 12, 13, 15, 0, 0, 0},
      {7, 11, 12, 14, 0, 0, 0}, {7, 11, 12, 14, 0, 0, 0}, {7, 11, 12, 14, 0, 0, 0}, {7, 11, 12, 7, 0, 0, 0},
      {0, 12, 10, 0, 0, 0, 0},  {0, 9, 10, 9, 10, 0, 0},  {16, 9, 9, 9, 11, 10, 0}, {17, 8, 9, 8, 9, 10, 0},
      {9, 7, 7, 7, 9, 6, 0},    {8, 7, 7, 6, 6, 6, 0},    {8, 6, 6, 6, 7, 7, 0},    {7, 6, 6, 6, 7, 6, 0},
      {7, 6, 6, 6, 6, 6, 0},    {6, 6, 6, 6, 6, 6, 0},    {6, 5, 6, 6, 6, 7, 10},   {9, 9, 9, 5, 5, 6, 6},
      {0, 0, 0, 0, 0, 0, 0}};
};

#endif
