#ifndef TrackProducer_SiStripClusterThinningProducer_h
#define TrackProducer_SiStripClusterThinningProducer_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/Framework/interface/stream/ThinningSelectorByRefBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterfwd.h"

namespace edm {
  class ParameterSetDescription;
}

class SiStripClusterSelector
    : public edm::ThinningSelectorByRefBase<edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> > {
public:
  SiStripClusterSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);
  static void fillDescription(edm::ParameterSetDescription& desc);
  void preChooseRefs(edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters,
                     edm::Event const& event,
                     edm::EventSetup const& es) override;

private:
  std::vector<edm::EDGetTokenT<TrackingRecHitCollection> > trackingRecHitsTokens_;
};

typedef edm::ThinningProducer<edmNew::DetSetVector<SiStripCluster>, SiStripClusterSelector>
    SiStripClusterThinningProducer;

#endif
