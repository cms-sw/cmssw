#ifndef SimHitCategory_h
#define SimHitCategory_h

namespace SimHitCategory {

  // Identification code of sim hit production mechanism, subdetector dependent

  // MTD

  static constexpr unsigned int n_mtd_categories = 5;

  static constexpr unsigned int mtd_prodType[n_mtd_categories] = {
      0,  // direct hit from particle coming from tracker
      1,  // BTL hit from secondary particle not saved in history
      2,  // BTL hit from identified looper
      3,  // BTL hit from back-scattering from CALO volume
      4   // ETL hit entering from a rear face of disks
  };

};  // namespace SimHitCategory

#endif
