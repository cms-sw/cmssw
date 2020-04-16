///////////////////////////////////////
//
// class Validation: Class to fill dqm monitor elements from existing EDM file
//
///////////////////////////////////////

#include "Validation/EventGenerator/interface/BPhysicsSpectrum.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;

BPhysicsSpectrum::BPhysicsSpectrum(const edm::ParameterSet& iPSet)
    : genparticleCollection_(iPSet.getParameter<edm::InputTag>("genparticleCollection")),
      // do not include weights right now to allow for running on aod
      name(iPSet.getParameter<std::string>("name")),
      mass_min(iPSet.getParameter<double>("massmin")),
      mass_max(iPSet.getParameter<double>("massmax")) {
  genparticleCollectionToken_ = consumes<reco::GenParticleCollection>(genparticleCollection_);
  Particles = iPSet.getParameter<std::vector<int> >("pdgids");
}

BPhysicsSpectrum::~BPhysicsSpectrum() {}

void BPhysicsSpectrum::bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) {
  DQMHelper dqm(&i);
  i.setCurrentFolder("Generator/BPhysics");
  Nobj = dqm.book1dHisto("NSpectrum" + name, "NSpectrum" + name, 1, 0., 1, "bin", "Number of " + name);
  mass = dqm.book1dHisto(name + "Mass", "Mass Spectrum", 100, mass_min, mass_max, "Mass (GeV)", "Number of Events");
}

void BPhysicsSpectrum::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genparticleCollectionToken_, genParticles);
  for (reco::GenParticleCollection::const_iterator iter = genParticles->begin(); iter != genParticles->end(); ++iter) {
    for (unsigned int i = 0; i < Particles.size(); i++) {
      if (abs(iter->pdgId()) == abs(Particles[i])) {
        Nobj->Fill(0.5, 1.0);
        mass->Fill(iter->mass(), 1.0);
      }
    }
  }
}
