#ifndef RecoTracker_PixelSeeding_interface_CAGeometry_h
#define RecoTracker_PixelSeeding_interface_CAGeometry_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"

namespace reco {

    // struct RZMap
    // {
    //   // in cm
    //   static constexpr float rmin = 0.f;
    //   static constexpr float rmax = 120.f;
    //   static constexpr float zlim = 300.f;
    //   static constexpr float zran = 600.f;

    //   static constexpr uint16_t binr = uint16_t(rmax) * 5; 
    //   static constexpr uint16_t binz = uint16_t(zlim) * 5; 

    //   static constexpr uint16_t binz = uint16_t(zlim) * 5;

    //   // bin = 1 + int (fNbins*(x-fXmin)/(fXmax-fXmin) );
    // }

    using GraphNode = std::array<uint32_t, 2>;
    using DetFrame = SOAFrame<float>;

    GENERATE_SOA_LAYOUT(CAModulesLayout, 
                    SOA_COLUMN(DetFrame, detFrame)
                    )

    GENERATE_SOA_LAYOUT(CALayersLayout, 
                    SOA_COLUMN(uint32_t, layerStarts),
                    SOA_COLUMN(float, caThetaCut),
                    SOA_COLUMN(float, caDCACut)
                    )

    GENERATE_SOA_LAYOUT(CAGraphLayout, 
                    SOA_COLUMN(GraphNode, graph),
                    SOA_COLUMN(bool, startingPair),
                    SOA_COLUMN(int16_t, phiCuts),
                    SOA_COLUMN(float, minz),
                    SOA_COLUMN(float, maxz),
                    SOA_COLUMN(float, maxr)
                    )
                    
  using CALayersSoA = CALayersLayout<>;
  using CALayersSoAView = CALayersSoA::View;
  using CALayersSoAConstView = CALayersSoA::ConstView;

  using CAGraphSoA = CAGraphLayout<>;
  using CAGraphSoAView = CAGraphSoA::View;
  using CAGraphSoAConstView = CAGraphSoA::ConstView;

  using CAModulesSoA = CAModulesLayout<>;
  using CAModulesView = CAModulesSoA::View;
  using CAModulesConstView = CAModulesSoA::ConstView;  

}
#endif  // RecoTracker_PixelSeeding_interface_CAGeometry_h