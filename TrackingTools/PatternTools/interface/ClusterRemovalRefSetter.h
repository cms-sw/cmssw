#ifndef TrackingTools_PatternTools_ClusterRemovalRefSetter_h
#define TrackingTools_PatternTools_ClusterRemovalRefSetter_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ClusterRemovalRefSetter {
public:
  inline ClusterRemovalRefSetter(const edm::Event& iEvent, const edm::InputTag& tag);
  inline ClusterRemovalRefSetter(const edm::Event& iEvent, const edm::EDGetTokenT<reco::ClusterRemovalInfo>& token);

  inline void reKey(TrackingRecHit* hit) const;

private:
  typedef OmniClusterRef::ClusterPixelRef ClusterPixelRef;
  typedef OmniClusterRef::ClusterStripRef ClusterStripRef;

  inline void reKeyPixel(OmniClusterRef& clusRef) const;
  inline void reKeyStrip(OmniClusterRef& clusRef) const;

private:
  const reco::ClusterRemovalInfo* cri_;
};

#include "ClusterRemovalRefSetter.icc"
#endif
