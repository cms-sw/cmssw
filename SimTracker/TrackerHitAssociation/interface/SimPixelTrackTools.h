#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace simpixeltracks {
  // function that determines the layerId from the detId for Phase 1 and 2
  template <typename TrackerTraits>
  unsigned int getLayerId(DetId const& detId, const TrackerTopology* trackerTopology) {
    // number of barrel layers
    constexpr unsigned int numBarrelLayers{4};
    // number of disks per endcap
    constexpr unsigned int numEndcapDisks = (TrackerTraits::numberOfLayers - numBarrelLayers) / 2;
    // number of pixel layers in total
    constexpr unsigned int numPixelLayers = (TrackerTraits::numberOfLayers);
    // number of OT barrel layers
    constexpr unsigned int numOTBarrelLayers{3};  // FIXME: hardcoded for now
    // number of disks per OT endcap
    constexpr unsigned int numOTEndcapDisks{5};  // FIXME: hardcoded for now

    // set default to 999 (invalid)
    unsigned int layerId{99};

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
          // add offest in the forward endcap to get, e.g. for Phase 2, from (1,12) to (4,15)
          layerId = trackerTopology->pxfDisk(detId) + numBarrelLayers - 1;
        }
        break;
      case SiStripSubdetector::TOB:
        // add offset in the OT barrel to get, e.g. for Phase 2, from (1,3) to (28, 30)
        layerId = trackerTopology->tobLayer(detId) + numPixelLayers - 1;
        break;
      case SiStripSubdetector::TID:
        if (trackerTopology->tidSide(detId) == 1) {
          // add offset in the OT forward endcap to get, e.g. for Phase 2, from (1,5) to (31, 35)
          layerId = trackerTopology->tidWheel(detId) + numPixelLayers + numOTBarrelLayers - 1;
        } else {
          // add offset in the OT backward endcap to get, e.g. for Phase 2, from (1,3) to (36, 40)
          layerId = trackerTopology->tidWheel(detId) + numPixelLayers + numOTBarrelLayers + numOTEndcapDisks - 1;
        }
        break;
    }
    // return the determined Id
    return layerId;
  }

  // function that determines the cluster size of a Pixel RecHit in local y direction
  // according to the formula used in Patatrack reconstruction
  int clusterYSize(OmniClusterRef::ClusterPixelRef const cluster, uint16_t const pixmx, int const maxCol) {
    // check if the cluster lies at the y-edge of the module
    if (cluster->minPixelCol() == 0 || cluster->maxPixelCol() == maxCol) {
      // if so, return -1
      return -1;
    }

    // column span (span of cluster in y direction)
    int span = cluster->colSpan();

    // total charge of the first and last column of digis respectively
    int q_firstCol = 0;
    int q_lastCol = 0;

    // loop over the pixels/digis of the cluster and update the charges of first and last column
    int offset;
    for (int i{0}; i < cluster->size(); i++) {
      offset = cluster->pixelOffset()[2 * i + 1];

      // check if pixel is in first column and eventually update the charge
      if (offset == 0) {
        q_firstCol += std::min(cluster->pixelADC()[i], pixmx);
      }
      // check if pixel is in last column and eventually update the charge
      if (offset == span) {
        q_lastCol += std::min(cluster->pixelADC()[i], pixmx);
      }
    }

    // calculate the unbalance term
    int unbalance = 8. * std::abs(float(q_firstCol - q_lastCol)) / float(q_firstCol + q_lastCol);

    // calculate the cluster size
    int clusterYSize = 8 * (span + 1) - unbalance;
    return clusterYSize;
  }
}  // namespace simpixeltracks