#ifndef RecoTracker_TrackProducer_ClusterRemovalRefSetter_h
#define RecoTracker_TrackProducer_ClusterRemovalRefSetter_h

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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ClusterRemovalRefSetter {
public:
  ClusterRemovalRefSetter(const edm::Event &iEvent, const edm::InputTag& tag) ;
  void reKey(TrackingRecHit *hit) const ;
private:
  typedef OmniClusterRef::ClusterPixelRef ClusterPixelRef;
  typedef OmniClusterRef::ClusterStripRef ClusterStripRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;

  void reKeyPixel(OmniClusterRef& clusRef) const ;
  void reKeyStrip(OmniClusterRef& clusRef) const ;
  //void reKeyRegional(OmniClusterRef& clusRef) const ;
private:
  const reco::ClusterRemovalInfo *cri_;
};

#endif
