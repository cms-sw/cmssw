#ifndef SimG4CMSForward_MtdHitCategory_h
#define SimG4CMSForward_MtdHitCategory_h

namespace MtdHitCategory {
  static constexpr unsigned int k_idsecOffset = 1;
  static constexpr unsigned int k_idloopOffset = 2;
  static constexpr unsigned int k_idFromCaloOffset = 3;
  static constexpr unsigned int k_idETLfromBack = 4;
  static constexpr unsigned int n_categories =
      std::max({k_idsecOffset, k_idloopOffset, k_idFromCaloOffset, k_idETLfromBack});
};  // namespace MtdHitCategory

#endif
