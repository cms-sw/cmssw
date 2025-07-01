#ifndef RecoTracker_PixelSeeding_interface_CACoupleSoA_h
#define RecoTracker_PixelSeeding_interface_CACoupleSoA_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace caStructures {

  GENERATE_SOA_LAYOUT(CACoupleLayout, SOA_COLUMN(uint32_t, inner), SOA_COLUMN(uint32_t, outer))

  using CACoupleSoA = CACoupleLayout<>;
  using CACoupleSoAView = CACoupleSoA::View;
  using CACoupleSoAConstView = CACoupleSoA::ConstView;

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_CACoupleSoA_h
