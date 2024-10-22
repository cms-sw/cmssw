#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace TopInitID {
  static constexpr int tID = 6;
}  // namespace TopInitID

class TopInitSubset : public edm::global::EDProducer<> {
public:
  explicit TopInitSubset(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void fillOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&) const;

private:
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
};

TopInitSubset::TopInitSubset(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))) {
  produces<reco::GenParticleCollection>();
}

void TopInitSubset::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const {
  edm::Handle<reco::GenParticleCollection> src;
  evt.getByToken(srcToken_, src);

  const reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>();
  auto sel = std::make_unique<reco::GenParticleCollection>();

  //fill output collection
  fillOutput(*src, *sel);

  evt.put(std::move(sel));
}

void TopInitSubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel) const {
  for (auto const& t : src) {
    if (std::abs(t.pdgId()) == TopInitID::tID) {
      bool hasTopMother = false;
      for (unsigned idx = 0; idx < t.numberOfMothers(); ++idx)
        if (std::abs(t.mother(idx)->pdgId()) == TopInitID::tID)
          hasTopMother = true;
      if (hasTopMother)
        continue;
      for (unsigned idx = 0; idx < t.numberOfMothers(); ++idx) {
        sel.emplace_back(t.mother(idx)->threeCharge(),
                         t.mother(idx)->p4(),
                         t.mother(idx)->vertex(),
                         t.mother(idx)->pdgId(),
                         t.mother(idx)->status(),
                         false);
      }
      break;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopInitSubset);
