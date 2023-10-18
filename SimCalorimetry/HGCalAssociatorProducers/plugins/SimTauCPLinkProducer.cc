// -*- C++ -*-
//
//
// Authors:  Marco Rovere, Andreas Gruber
//         Created:  Mon, 16 Oct 2023 14:24:35 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimTauCPLink.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class SimTauProducer : public edm::stream::EDProducer<> {
public:
  explicit SimTauProducer(const edm::ParameterSet&);
  ~SimTauProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void buildSimTau(SimTauCPLink&,
                   uint8_t,
                   int,
                   const reco::GenParticle&,
                   int,
                   edm::Handle<std::vector<CaloParticle>>,
                   const std::vector<int>&);
  int buildDecayModes(const SimTauCPLink&);
  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticle_token_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_token_;
  const edm::EDGetTokenT<std::vector<int>> genBarcodes_token_;
};

SimTauProducer::SimTauProducer(const edm::ParameterSet& iConfig)
    : caloParticle_token_(consumes<std::vector<CaloParticle>>(iConfig.getParameter<edm::InputTag>("CaloParticle"))),
      genParticles_token_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("GenParticles"))),
      genBarcodes_token_(consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("GenBarcodes"))) {
  produces<std::vector<SimTauCPLink>>();
}

void SimTauProducer::buildSimTau(SimTauCPLink& t,
                                 uint8_t generation,
                                 int resonance_idx,
                                 const reco::GenParticle& gen_particle,
                                 int gen_particle_key,
                                 edm::Handle<std::vector<CaloParticle>> calo_particle_h,
                                 const std::vector<int>& gen_particle_barcodes) {
  const auto& caloPartVec = *calo_particle_h;
  auto& daughters = gen_particle.daughterRefVector();
  bool is_leaf = (daughters.empty());
  if (is_leaf) {
    LogDebug("SimTauProducer") << " TO BE SAVED " + std::to_string(resonance_idx) + " ";
    auto const& gen_particle_barcode = gen_particle_barcodes[gen_particle_key];
    auto const& found_in_caloparticles = std::find_if(caloPartVec.begin(), caloPartVec.end(), [&](const auto& p) {
      return p.g4Tracks()[0].genpartIndex() == gen_particle_barcode;
    });
    if (found_in_caloparticles != caloPartVec.end()) {
      auto calo_particle_idx = (found_in_caloparticles - caloPartVec.begin());
      t.calo_particle_leaves.push_back(CaloParticleRef(calo_particle_h, calo_particle_idx));
      t.leaves.push_back(
          {gen_particle.pdgId(), resonance_idx, (int)t.calo_particle_leaves.size() - 1, gen_particle_key});
      LogDebug("SimTauProducer") << " CP " + std::to_string(calo_particle_idx) + " " << caloPartVec[calo_particle_idx];
    } else {
      t.leaves.push_back({gen_particle.pdgId(), resonance_idx, -1, gen_particle_key});
    }
    return;
  } else if (generation != 0) {
    t.resonances.push_back({gen_particle.pdgId(), resonance_idx});
    resonance_idx = t.resonances.size() - 1;
    LogDebug("SimTauProducer") << " RESONANCE/INTERMEDIATE " + std::to_string(resonance_idx) + " ";
  }

  ++generation;
  for (auto daughter = daughters.begin(); daughter != daughters.end(); ++daughter) {
    int gen_particle_key = (*daughter).key();
    LogDebug("SimTauProducer") << " gen " + std::to_string((int)generation) + " " + std::to_string(gen_particle_key) +
                                      " " + std::to_string((*daughter)->pdgId()) + " ";
    buildSimTau(t, generation, resonance_idx, *(*daughter), gen_particle_key, calo_particle_h, gen_particle_barcodes);
  }
}

int SimTauProducer::buildDecayModes(const SimTauCPLink& t) {
  enum decayModes {
    kNull = -1,
    kOneProng0PiZero,
    kOneProng1PiZero,
    kOneProng2PiZero,
    kOneProng3PiZero,
    kOneProngNPiZero,
    kTwoProng0PiZero,
    kTwoProng1PiZero,
    kTwoProng2PiZero,
    kTwoProng3PiZero,
    kTwoProngNPiZero,
    kThreeProng0PiZero,
    kThreeProng1PiZero,
    kThreeProng2PiZero,
    kThreeProng3PiZero,
    kThreeProngNPiZero,
    kRareDecayMode,
    kElectron,
    kMuon
  };

  int numElectrons = 0;
  int numMuons = 0;
  int numHadrons = 0;
  int numPhotons = 0;
  auto& leaves = t.leaves;
  for (auto leaf : leaves) {
    int pdg_id = abs(leaf.pdgId());
    switch (pdg_id) {
      case 22:
        numPhotons++;
        break;
      case 11:
        numElectrons++;
        break;
      case 13:
        numMuons++;
        break;
      case 16:
        break;
      default:
        numHadrons++;
    }
  }

  if (numElectrons == 1)
    return kElectron;
  else if (numMuons == 1)
    return kMuon;
  switch (numHadrons) {
    case 1:
      switch (numPhotons) {
        case 0:
          return kOneProng0PiZero;
        case 2:
          return kOneProng1PiZero;
        case 4:
          return kOneProng2PiZero;
        case 6:
          return kOneProng3PiZero;
        default:
          return kOneProngNPiZero;
      }
    case 2:
      switch (numPhotons) {
        case 0:
          return kTwoProng0PiZero;
        case 2:
          return kTwoProng1PiZero;
        case 4:
          return kTwoProng2PiZero;
        case 6:
          return kTwoProng3PiZero;
        default:
          return kTwoProngNPiZero;
      }
    case 3:
      switch (numPhotons) {
        case 0:
          return kThreeProng0PiZero;
        case 2:
          return kThreeProng1PiZero;
        case 4:
          return kThreeProng2PiZero;
        case 6:
          return kThreeProng3PiZero;
        default:
          return kThreeProngNPiZero;
      }
    default:
      return kRareDecayMode;
  }
}

// ------------ method called for each event  ------------
void SimTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<std::vector<CaloParticle>> CaloParticle_h;
  iEvent.getByToken(caloParticle_token_, CaloParticle_h);

  edm::Handle<std::vector<reco::GenParticle>> gen_particles_h;
  iEvent.getByToken(genParticles_token_, gen_particles_h);

  Handle<std::vector<int>> gen_barcodes_h;
  iEvent.getByToken(genBarcodes_token_, gen_barcodes_h);

  const auto& genParticles = *gen_particles_h;
  const auto& genBarcodes = *gen_barcodes_h;
  auto tauDecayVec = std::make_unique<std::vector<SimTauCPLink>>();
  for (auto const& g : genParticles) {
    auto const& flags = g.statusFlags();
    if (std::abs(g.pdgId()) == 15 and flags.isPrompt() and flags.isDecayedLeptonHadron()) {
      SimTauCPLink t;
      buildSimTau(t, 0, -1, g, -1, CaloParticle_h, genBarcodes);
      t.decayMode = buildDecayModes(t);
      t.dump();
      t.dumpFullDecay();
      (*tauDecayVec).push_back(t);
    }
  }
  iEvent.put(std::move(tauDecayVec));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SimTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimTauProducer);
