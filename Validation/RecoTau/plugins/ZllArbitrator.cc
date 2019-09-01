#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <algorithm>
#include <memory>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
class ZllArbitrator : public edm::global::EDProducer<> {
public:
  explicit ZllArbitrator(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  edm::EDGetTokenT<std::vector<reco::CompositeCandidate>> srcZCand_;
};

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

ZllArbitrator::ZllArbitrator(edm::ParameterSet const& iConfig)
    : srcZCand_{consumes<std::vector<reco::CompositeCandidate>>(
          iConfig.getParameter<edm::InputTag>("ZCandidateCollection"))} {
  produces<std::vector<reco::CompositeCandidate>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void ZllArbitrator::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  edm::Handle<std::vector<reco::CompositeCandidate>> zCandidates;
  iEvent.getByToken(srcZCand_, zCandidates);

  auto bestZ = std::make_unique<std::vector<reco::CompositeCandidate>>();
  if (!zCandidates->empty()) {
    // If you're going to hard-code numbers, at least make them constexpr.
    double constexpr ZmassPDG{91.18};  // GeV

    auto bestZCand = std::min_element(
        std::cbegin(*zCandidates), std::cend(*zCandidates), [ZmassPDG](auto const& firstCand, auto const& secondCand) {
          return std::abs(firstCand.mass() - ZmassPDG) < std::abs(secondCand.mass() - ZmassPDG);
        });
    bestZ->push_back(*bestZCand);
  }

  iEvent.put(std::move(bestZ));
}

using BestMassZArbitrationProducer = ZllArbitrator;

DEFINE_FWK_MODULE(BestMassZArbitrationProducer);
