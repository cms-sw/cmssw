#ifndef SimTracker_TrackAssociation_trackHitsToClusterRefs_h
#define SimTracker_TrackAssociation_trackHitsToClusterRefs_h

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace track_associator {
  inline const TrackingRecHit *getHitFromIter(trackingRecHit_iterator iter) { return &(**iter); }

  inline const TrackingRecHit *getHitFromIter(TrackingRecHitCollection::const_iterator iter) { return &(*iter); }

  template <typename iter>
  std::vector<OmniClusterRef> hitsToClusterRefs(iter begin, iter end) {
    std::vector<OmniClusterRef> returnValue;
    for (iter iRecHit = begin; iRecHit != end; ++iRecHit) {
      const TrackingRecHit *rhit = getHitFromIter(iRecHit);
      if (trackerHitRTTI::isFromDet(*rhit)) {
        int subdetid = rhit->geographicalId().subdetId();
        if (subdetid == PixelSubdetector::PixelBarrel || subdetid == PixelSubdetector::PixelEndcap) {
          const SiPixelRecHit *pRHit = dynamic_cast<const SiPixelRecHit *>(rhit);
          if (pRHit && !pRHit->cluster().isNonnull())
            edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                             << " file: " << __FILE__ << " line: " << __LINE__;
          returnValue.push_back(pRHit->omniClusterRef());
        } else if (subdetid == SiStripDetId::TIB || subdetid == SiStripDetId::TOB || subdetid == SiStripDetId::TID ||
                   subdetid == SiStripDetId::TEC) {
          const std::type_info &tid = typeid(*rhit);
          if (tid == typeid(SiStripMatchedRecHit2D)) {
            const SiStripMatchedRecHit2D *sMatchedRHit = dynamic_cast<const SiStripMatchedRecHit2D *>(rhit);
            if (!sMatchedRHit->monoHit().cluster().isNonnull() || !sMatchedRHit->stereoHit().cluster().isNonnull())
              edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                               << " file: " << __FILE__ << " line: " << __LINE__;
            returnValue.push_back(sMatchedRHit->monoClusterRef());
            returnValue.push_back(sMatchedRHit->stereoClusterRef());
          } else if (tid == typeid(SiStripRecHit2D)) {
            const SiStripRecHit2D *sRHit = dynamic_cast<const SiStripRecHit2D *>(rhit);
            if (!sRHit->cluster().isNonnull())
              edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                               << " file: " << __FILE__ << " line: " << __LINE__;
            returnValue.push_back(sRHit->omniClusterRef());
          } else if (tid == typeid(SiStripRecHit1D)) {
            const SiStripRecHit1D *sRHit = dynamic_cast<const SiStripRecHit1D *>(rhit);
            if (!sRHit->cluster().isNonnull())
              edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                               << " file: " << __FILE__ << " line: " << __LINE__;
            returnValue.push_back(sRHit->omniClusterRef());
          } else if (tid == typeid(Phase2TrackerRecHit1D)) {
            const Phase2TrackerRecHit1D *ph2Hit = dynamic_cast<const Phase2TrackerRecHit1D *>(rhit);
            if (!ph2Hit->cluster().isNonnull())
              edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                               << " file: " << __FILE__ << " line: " << __LINE__;
            returnValue.push_back(ph2Hit->omniClusterRef());
          } else if (tid == typeid(VectorHit)) {
            const VectorHit *vectorHit = dynamic_cast<const VectorHit *>(rhit);
            if (!vectorHit->cluster().isNonnull())
              edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!"
                                               << " file: " << __FILE__ << " line: " << __LINE__;
            returnValue.push_back(vectorHit->firstClusterRef());

          } else {
            auto const &thit = static_cast<BaseTrackerRecHit const &>(*rhit);
            if (thit.isProjected()) {
            } else {
              edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to "
                                                  "any SiStripCluster! subdetid = "
                                               << subdetid;
            }
          }
        } else {
          edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to any "
                                              "cluster! subdetid = "
                                           << subdetid;
        }
      }
    }
    return returnValue;
  }
}  // namespace track_associator

#endif
