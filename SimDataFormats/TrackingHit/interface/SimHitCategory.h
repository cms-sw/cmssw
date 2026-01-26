#ifndef SimHitCategory_h
#define SimHitCategory_h

// 7 bits field availablen PSimHit processType, i.e. up 127
static constexpr unsigned int k_procidMask_ = 0x1FF;
static constexpr unsigned int k_hitidMask_ = 0x7F;
static constexpr unsigned int k_hitidShift_ = 9;

namespace SimHitCategory {

  // Identification code of sim hit production mechanism, subdetector dependent

  static constexpr unsigned int k_BTLsecondary = 1;  // BTL hit from secondary particle not saved in history
  static constexpr unsigned int k_BTLlooper = 2;     // BTL hit from identified looper
  static constexpr unsigned int k_BTLfromCalo = 3;   // BTL hit from back-scattering from CALO volume
  static constexpr unsigned int k_ETLfromBack = 4;   // ETL hit entering from a rear face of disks

};  // namespace SimHitCategory

#endif
