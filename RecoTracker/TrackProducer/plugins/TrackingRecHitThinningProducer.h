#ifndef TrackProducer_TrackingRecHitThinningProducer_h
#define TrackProducer_TrackingRecHitThinningProducer_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/Framework/interface/stream/ThinningSelectorByRefBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

namespace edm {
  class ParameterSetDescription;
}

class TrackingRecHitSelector : public edm::ThinningSelectorByRefBase<edm::Ref<TrackingRecHitCollection> > {
public:
  TrackingRecHitSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);
  static void fillPSetDescription(edm::ParameterSetDescription& desc);
  void preChooseRefs(edm::Handle<TrackingRecHitCollection> hits,
                     edm::Event const& event,
                     edm::EventSetup const& es) override;

private:
  edm::EDGetTokenT<reco::TrackExtraCollection> trackExtraToken_;
};

typedef edm::ThinningProducer<TrackingRecHitCollection, TrackingRecHitSelector> TrackingRecHitThinningProducer;

#endif
