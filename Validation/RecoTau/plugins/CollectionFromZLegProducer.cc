#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class CollectionFromZLegProducer : public edm::global::EDProducer<> {
public:
  explicit CollectionFromZLegProducer(edm::ParameterSet const& iConfig);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  edm::EDGetTokenT<std::vector<reco::CompositeCandidate>> v_RecoCompositeCandidateToken_;
};

////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

CollectionFromZLegProducer::CollectionFromZLegProducer(edm::ParameterSet const& iConfig)
    : v_RecoCompositeCandidateToken_{consumes<std::vector<reco::CompositeCandidate>>(
          iConfig.getParameter<edm::InputTag>("ZCandidateCollection"))} {
  produces<std::vector<reco::CompositeCandidate>>("theTagLeg");
  produces<std::vector<reco::CompositeCandidate>>("theProbeLeg");
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void CollectionFromZLegProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto tagLegs = std::make_unique<std::vector<reco::CompositeCandidate>>();
  auto probeLegs = std::make_unique<std::vector<reco::CompositeCandidate>>();

  edm::Handle<std::vector<reco::CompositeCandidate>> zs;
  iEvent.getByToken(v_RecoCompositeCandidateToken_, zs);

  // this is specific for our 'tag and probe'
  for (auto const& z : *zs) {
    int c{};
    for (auto const& leg : z) {
      if (c == 0) {
        tagLegs->emplace_back(leg);
      } else if (c == 1) {
        probeLegs->emplace_back(leg);
      } else {
        break;
      }
      ++c;
    }
  }
  iEvent.put(std::move(tagLegs), "theTagLeg");
  iEvent.put(std::move(probeLegs), "theProbeLeg");
}

DEFINE_FWK_MODULE(CollectionFromZLegProducer);
