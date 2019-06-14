#include "TopQuarkAnalysis/TopEventProducers/interface/TopInitSubset.h"

TopInitSubset::TopInitSubset(const edm::ParameterSet& cfg)
    : srcToken_(consumes<reco::GenParticleCollection>(cfg.getParameter<edm::InputTag>("src"))) {
  produces<reco::GenParticleCollection>();
}

TopInitSubset::~TopInitSubset() {}

void TopInitSubset::produce(edm::Event& evt, const edm::EventSetup& setup) {
  edm::Handle<reco::GenParticleCollection> src;
  evt.getByToken(srcToken_, src);

  const reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>();
  std::unique_ptr<reco::GenParticleCollection> sel(new reco::GenParticleCollection);

  //fill output collection
  fillOutput(*src, *sel);

  evt.put(std::move(sel));
}

void TopInitSubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel) {
  for (reco::GenParticleCollection::const_iterator t = src.begin(); t != src.end(); ++t) {
    if (std::abs(t->pdgId()) == TopInitID::tID) {
      bool hasTopMother = false;
      for (unsigned idx = 0; idx < t->numberOfMothers(); ++idx)
        if (std::abs(t->mother(idx)->pdgId()) == TopInitID::tID)
          hasTopMother = true;
      if (hasTopMother)
        continue;
      for (unsigned idx = 0; idx < t->numberOfMothers(); ++idx) {
        reco::GenParticle* cand = new reco::GenParticle(t->mother(idx)->threeCharge(),
                                                        t->mother(idx)->p4(),
                                                        t->mother(idx)->vertex(),
                                                        t->mother(idx)->pdgId(),
                                                        t->mother(idx)->status(),
                                                        false);
        std::unique_ptr<reco::GenParticle> ptr(cand);
        sel.push_back(*ptr);
      }
      break;
    }
  }
}
