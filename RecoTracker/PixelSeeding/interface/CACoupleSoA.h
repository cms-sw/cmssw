#ifndef RecoTracker_PixelSeeding_interface_CACoupleSoA_h
#define RecoTracker_PixelSeeding_interface_CACoupleSoA_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace caStructures {

  // May be used for:
  // - cells (couples of hits)
  // - triplets (couples of cells)
  
  GENERATE_SOA_LAYOUT(CACouple,
                        SOA_COLUMN(uint32_t, inner),
                        SOA_COLUMN(uint32_t, outer),
                        )
                    
  using CACoupleSoA = CACoupleLayout<>;
  using CACoupleSoAView = CACoupleSoA::View;
  using CACoupleSoAConstView = CACoupleSoA::ConstView;

}
#endif  // RecoTracker_PixelSeeding_interface_CACoupleSoA_h