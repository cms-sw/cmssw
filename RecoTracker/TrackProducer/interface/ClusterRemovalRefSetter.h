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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class ClusterRemovalRefSetter {
    public:
        ClusterRemovalRefSetter(const edm::Event &iEvent, const edm::InputTag tag) ;
        void reKey(TrackingRecHit *hit) const ;
        void reKey(SiStripRecHit2D *hit, uint32_t detid) const ;
        void reKey(SiPixelRecHit *hit, uint32_t detid) const ;
    private:
        const reco::ClusterRemovalInfo *cri_;
        edm::Handle<edmNew::DetSetVector<SiStripCluster> > handleStrip_; // note: Handles are not const,
        edm::Handle<edmNew::DetSetVector<SiPixelCluster> > handlePixel_; // but they wrap const pointers.
};

#endif
