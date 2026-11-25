#ifndef RecoTracker_PixelSeeding_test_models_TrackerLayerId_h
#define RecoTracker_PixelSeeding_test_models_TrackerLayerId_h

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

namespace tracknano {

  // Flat layer numbering of the whole tracker, used as the "layerId" column of the training
  // nano tables: pixel barrel 0-3, pixel forward/backward disks 4-27, then the outer tracker
  // (barrel 28-33, forward/backward TID wheels 34-53, PS and SS interleaved).
  template <typename TrackerTraits, typename IntType>
  IntType getLayerId(DetId const& detId,
                     const TrackerTopology* trackerTopology,
                     const TrackerGeometry* trackerGeometry = nullptr) {
    constexpr IntType numBarrelLayers{4};
    constexpr IntType numEndcapDisks = (TrackerTraits::numberOfLayers - numBarrelLayers) / 2;
    constexpr IntType numPixelLayers = (TrackerTraits::numberOfLayers);
    constexpr IntType numOTBarrelLayers{6};
    constexpr IntType numOTEndcapDisks{5};
    // 99 = invalid
    IntType layerId{99};

    switch (detId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        // subtract 1 in the barrel to get, e.g. for Phase 2, from (1,4) to (0,3)
        layerId = trackerTopology->pxbLayer(detId) - 1;
        break;
      case PixelSubdetector::PixelEndcap:
        if (trackerTopology->pxfSide(detId) == 1) {
          // add offset in the backward endcap to get, e.g. for Phase 2, from (1,12) to (16,27)
          layerId = trackerTopology->pxfDisk(detId) + numBarrelLayers + numEndcapDisks - 1;
        } else {
          // add offset in the forward endcap to get, e.g. for Phase 2, from (1,12) to (4,15)
          layerId = trackerTopology->pxfDisk(detId) + numBarrelLayers - 1;
        }
        break;
      case SiStripSubdetector::TOB:
        // add offset in the OT barrel to get, e.g. for Phase 2, from (1,6) to (28,33)
        layerId = trackerTopology->tobLayer(detId) + numPixelLayers - 1;
        break;
      case SiStripSubdetector::TID: {
        bool isPS = trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
        if (trackerTopology->tidSide(detId) == 2) {
          // add offset in the OT forward endcap to get, e.g. for Phase 2, from (1,5) to (34,43)
          layerId = 2 * (trackerTopology->tidWheel(detId) - 1) + numPixelLayers + numOTBarrelLayers + (isPS ? 0 : 1);
        } else {
          // add offset in the OT backward endcap to get, e.g. for Phase 2, from (1,5) to (44,53)
          layerId = 2 * (trackerTopology->tidWheel(detId) - 1 + numOTEndcapDisks) + numPixelLayers + numOTBarrelLayers +
                    (isPS ? 0 : 1);
        }
      } break;
    }
    return layerId;
  }

}  // namespace tracknano

#endif
