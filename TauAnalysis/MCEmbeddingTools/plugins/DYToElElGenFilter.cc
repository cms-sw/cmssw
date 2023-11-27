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

class DYToElElGenFilter : public edm::stream::EDFilter<> {
public:
  explicit DYToElElGenFilter(const edm::ParameterSet &);
  ~DYToElElGenFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginStream(edm::StreamID) override;
  bool filter(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;

  edm::InputTag inputTag_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleCollection_;

  edm::Handle<reco::GenParticleCollection> gen_handle;

  // virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  // virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  // virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  // virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
};

DYToElElGenFilter::DYToElElGenFilter(const edm::ParameterSet &iConfig) {
  inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");
  genParticleCollection_ = consumes<reco::GenParticleCollection>(inputTag_);
}

DYToElElGenFilter::~DYToElElGenFilter() {}

bool DYToElElGenFilter::filter(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  iEvent.getByToken(genParticleCollection_, gen_handle);

  for (unsigned int i = 0; i < gen_handle->size(); i++) {
    const reco::GenParticle gen_particle = (*gen_handle)[i];
    // Check if Z Boson decayed into two leptons
    if (gen_particle.pdgId() == 23 && gen_particle.numberOfDaughters() == 2) {
      // Check if daugther particles are Electrons
      if (std::abs(gen_particle.daughter(0)->pdgId()) == 11 && std::abs(gen_particle.daughter(1)->pdgId()) == 11 &&
          std::abs(gen_particle.daughter(0)->eta()) < 2.6 && std::abs(gen_particle.daughter(1)->eta()) < 2.6 &&
          gen_particle.daughter(0)->pt() > 7 && gen_particle.daughter(1)->pt() > 7) {
        edm::LogPrint("") << "Electron Event ! ";
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}
// ------------ method called once each stream before processing any runs, lumis or events  ------------
void DYToElElGenFilter::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void DYToElElGenFilter::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DYToElElGenFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  //  Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DYToElElGenFilter);
