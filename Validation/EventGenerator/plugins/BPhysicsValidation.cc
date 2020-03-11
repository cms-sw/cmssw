///////////////////////////////////////
//
// class Validation: Class to fill dqm monitor elements from existing EDM file
//
///////////////////////////////////////

#include "Validation/EventGenerator/interface/BPhysicsValidation.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;

BPhysicsValidation::BPhysicsValidation(const edm::ParameterSet& iPSet)
    : genparticleCollection_(iPSet.getParameter<edm::InputTag>("genparticleCollection")),
      // do not include weights right now to allow for running on aod
      name(iPSet.getParameter<std::string>("name")),
      particle(name, iPSet) {
  genparticleCollectionToken_ = consumes<reco::GenParticleCollection>(genparticleCollection_);
  std::vector<std::string> daughterNames = iPSet.getParameter<std::vector<std::string> >("daughters");
  for (unsigned int i = 0; i < daughterNames.size(); i++) {
    std::string curSet = daughterNames[i];
    daughters.push_back(ParticleMonitor(name + curSet, iPSet.getUntrackedParameter<ParameterSet>(curSet)));
  }
}

BPhysicsValidation::~BPhysicsValidation() {}

void BPhysicsValidation::bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) {
  DQMHelper dqm(&i);
  i.setCurrentFolder("Generator/BPhysics");
  Nobj = dqm.book1dHisto("N" + name, "N" + name, 1, 0., 1, "bin", "Number of " + name);
  particle.Configure(i);
  for (unsigned int j = 0; j < daughters.size(); j++) {
    daughters[j].Configure(i);
  }
}

void BPhysicsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genparticleCollectionToken_, genParticles);
  for (reco::GenParticleCollection::const_iterator iter = genParticles->begin(); iter != genParticles->end(); ++iter) {
    if (abs(iter->pdgId()) == abs(particle.PDGID())) {
      Nobj->Fill(0.5, 1.0);
      particle.Fill(&(*iter), 1.0);
      FillDaughters(&(*iter));
    }
  }
}

void BPhysicsValidation::FillDaughters(const reco::GenParticle* p) {
  int mpdgid = p->pdgId();
  for (unsigned int i = 0; i < p->numberOfDaughters(); i++) {
    const reco::GenParticle* dau = static_cast<const reco::GenParticle*>(p->daughter(i));
    int pdgid = dau->pdgId();
    for (unsigned int i = 0; i < daughters.size(); i++) {
      if (abs(mpdgid) != abs(daughters[i].PDGID()) && daughters[i].PDGID() == pdgid)
        daughters[i].Fill(dau, 1.0);
      // note: use abs when comparing to mother to avoid mixing
    }
    FillDaughters(dau);
  }
}
