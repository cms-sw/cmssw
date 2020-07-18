#ifndef TrackProducer_SiPixelClusterThinningProducer_h
#define TrackProducer_SiPixelClusterThinningProducer_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/Framework/interface/stream/ThinningSelectorByRefBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

class SiPixelCluster;

namespace edm {
  class ParameterSetDescription;
}

class SiPixelClusterSelector
    : public edm::ThinningSelectorByRefBase<edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> > {
public:
  SiPixelClusterSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);
  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  void preChooseRefs(edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusters,
                     edm::Event const& event,
                     edm::EventSetup const& es) override;

private:
  std::vector<edm::EDGetTokenT<TrackingRecHitCollection> > trackingRecHitsTokens_;
};

typedef edm::ThinningProducer<edmNew::DetSetVector<SiPixelCluster>, SiPixelClusterSelector>
    SiPixelClusterThinningProducer;

#endif
