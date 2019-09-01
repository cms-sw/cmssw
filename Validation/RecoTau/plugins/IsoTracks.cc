#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class IsoTracks : public edm::global::EDProducer<> {
public:
  explicit IsoTracks(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  double coneRadius_;
  double threshold_;
  edm::EDGetTokenT<std::vector<reco::Track>> v_recoTrackToken_;
};

////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

IsoTracks::IsoTracks(edm::ParameterSet const& iConfig)
    : coneRadius_{iConfig.getParameter<double>("radius")},
      threshold_{iConfig.getParameter<double>("SumPtFraction")},
      v_recoTrackToken_{consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("src"))} {
  produces<std::vector<reco::Track>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void IsoTracks::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto isoTracks = std::make_unique<std::vector<reco::Track>>();

  edm::Handle<std::vector<reco::Track>> dirtyTracks;
  iEvent.getByToken(v_recoTrackToken_, dirtyTracks);

  if (dirtyTracks->empty()) {
    iEvent.put(std::move(isoTracks));
    return;
  }

  double sumPtInCone{};
  for (auto it1 = dirtyTracks->begin(); it1 != dirtyTracks->end(); ++it1) {
    for (auto it2 = dirtyTracks->begin(); it2 != dirtyTracks->end(); ++it2) {
      if (it1 == it2)
        continue;
      if (deltaR(it1->eta(), it1->phi(), it2->eta(), it2->phi()) < coneRadius_) {
        sumPtInCone += it2->pt();
      }
    }
    if (sumPtInCone <= threshold_ * it1->pt()) {
      isoTracks->push_back(*it1);
    }
  }

  iEvent.put(std::move(isoTracks));
}

DEFINE_FWK_MODULE(IsoTracks);
