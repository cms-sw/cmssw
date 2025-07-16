#ifndef RecoTracker_PixelSeeding_interface_CAPairSoA_h
#define RecoTracker_PixelSeeding_interface_CAPairSoA_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace caStructures {

  GENERATE_SOA_LAYOUT(CAPairLayout, SOA_COLUMN(uint32_t, inner), SOA_COLUMN(uint32_t, outer))

  using CAPairSoA = CAPairLayout<>;
  using CAPairSoAView = CAPairSoA::View;
  using CAPairSoAConstView = CAPairSoA::ConstView;

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_CAPairSoA_h
