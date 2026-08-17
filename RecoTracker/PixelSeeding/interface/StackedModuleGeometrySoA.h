#ifndef RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h
#define RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h

#include <cstdint>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // Phase-2 OT stacked-module geometry for stub formation: two closely-spaced sensors per module.
  // PS (Pixel-Strip): inner=macro-pixel, outer=strip, precise z; SS (Strip-Strip): no precise z.
  GENERATE_SOA_LAYOUT(
      StackedModuleGeometryLayout,
      SOA_COLUMN(uint32_t, detId),
      SOA_COLUMN(uint32_t, stackedDetId),
      // 0 = PSP, 1 = PSS, 2 = SS
      SOA_COLUMN(uint8_t, moduleType),
      // Distance between the inner and outer sensors (mm)
      SOA_COLUMN(float, sensorSeparation),

      // Tilt angle (rad) of the module axis from the radial direction: atan2(dz_phys, dr_phys) along the
      // physical inner->outer direction, so that dPhiDr has the same sign for flipped and non-flipped
      // modules. Flat barrel ~0, endcap ~+/-pi/2.
      SOA_COLUMN(float, tiltAngle),

      // sin/cos of tiltAngle, for dr_effective = separation / (cosTilt + sinTilt * z/r)
      SOA_COLUMN(float, sinTilt),
      SOA_COLUMN(float, cosTilt),

      // true for PS (pixel-strip), false for SS (strip-strip)
      SOA_COLUMN(bool, isPS),

      // topological "lower" sensor is physically farther from the beam line
      SOA_COLUMN(bool, isFlipped),

      // true for barrel, false for endcap
      SOA_COLUMN(bool, isBarrel),

      // module axis nearly radial, |cos(tiltAngle)| > cos(0.1); only meaningful in the barrel
      SOA_COLUMN(bool, isFlat),

      // true for z > 0; only meaningful in the endcap
      SOA_COLUMN(bool, isFwdEndcap),

      // OT layer number (0-5)
      SOA_COLUMN(uint8_t, layer),

      // Unit physical inner->outer sensor vector in global coordinates, consistent with tiltAngle
      SOA_COLUMN(float, globalLowUpNormX),
      SOA_COLUMN(float, globalLowUpNormY),
      SOA_COLUMN(float, globalLowUpNormZ),

      // Local x-axis direction in global coordinates, used by the parallax correction
      SOA_COLUMN(float, localXInGlobalX),
      SOA_COLUMN(float, localXInGlobalY),
      SOA_COLUMN(float, localXInGlobalZ),

      // Sensor surface frames (position + rotation), used by the stub bend-error formula and as the source
      // of CAModulesSoA::innerSensorFrame
      SOA_COLUMN(SOAFrame<float>, lowerSensorFrame),
      SOA_COLUMN(SOAFrame<float>, upperSensorFrame))

  using StackedModuleGeometrySoA = StackedModuleGeometryLayout<>;
  using StackedModuleGeometryView = StackedModuleGeometrySoA::View;
  using StackedModuleGeometryConstView = StackedModuleGeometrySoA::ConstView;

}  // namespace reco

#endif  // RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h
