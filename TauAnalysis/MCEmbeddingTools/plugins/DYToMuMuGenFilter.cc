#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DYToMuMuGenFilter : public edm::stream::EDFilter<> {
public:
  explicit DYToMuMuGenFilter(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool filter(edm::Event &, const edm::EventSetup &) override;

  edm::InputTag inputTag_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleCollection_;

  edm::Handle<reco::GenParticleCollection> gen_handle;

  // ----------member data ---------------------------
};

DYToMuMuGenFilter::DYToMuMuGenFilter(const edm::ParameterSet &iConfig) {
  inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");
  genParticleCollection_ = consumes<reco::GenParticleCollection>(inputTag_);
}

bool DYToMuMuGenFilter::filter(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  iEvent.getByToken(genParticleCollection_, gen_handle);

  for (unsigned int i = 0; i < gen_handle->size(); i++) {
    const reco::GenParticle gen_particle = (*gen_handle)[i];
    // Check if Z Boson decayed into two leptons
    if (gen_particle.pdgId() == 23 && gen_particle.numberOfDaughters() == 2) {
      // Debug output
      // std::cout << "pdgId" << gen_particle.pdgId() << std::endl;
      // std::cout << "nDau" << gen_particle.numberOfDaughters() << std::endl;
      // std::cout << "Dau1" << gen_particle->daughters->at(0).pdgId() << std::endl;
      // std::cout << "Dau2" << gen_particle.numberOfDaughters() << std::endl;
      // std::cout << "Dau1 " << gen_particle.daughter(0)->pdgId() << std::endl;
      // std::cout << "Dau2 " << gen_particle.daughter(1)->pdgId() << std::endl;
      // std::cout << gen_particle.daughter(1)->pdgId()+gen_particle.daughter(0)->pdgId() << std::endl;

      // Check if daugther particles are muons
      if (std::abs(gen_particle.daughter(0)->pdgId()) == 13 && std::abs(gen_particle.daughter(0)->eta()) < 2.6 &&
          std::abs(gen_particle.daughter(1)->eta()) < 2.6 && gen_particle.daughter(0)->pt() > 7 &&
          gen_particle.daughter(1)->pt() > 7) {
        // std::cout << "pdgId" << gen_particle.pdgId() << std::endl;
        // std::cout << "nDau" << gen_particle.numberOfDaughters() << std::endl;
        // std::cout << "Dau1 " << gen_particle.daughter(0)->pdgId() << std::endl;
        // std::cout << "Dau1 pt " << gen_particle.daughter(0)->pt() << std::endl;
        // std::cout << "Dau1 pt " << gen_particle.daughter(0)->eta() << std::endl;
        // std::cout << "Dau2 " << gen_particle.daughter(1)->pdgId() << std::endl;
        // std::cout << "Dau2 pt " << gen_particle.daughter(1)->pt() << std::endl;
        // std::cout << "Dau2 pt " << gen_particle.daughter(1)->eta() << std::endl;
        // std::cout << gen_particle.daughter(1)->pdgId()+gen_particle.daughter(0)->pdgId() << std::endl;
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DYToMuMuGenFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  //  Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DYToMuMuGenFilter);
