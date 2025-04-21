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
#include "FWCore/Framework/interface/global/EDProducer.h"

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

class SimTauProducer : public edm::global::EDProducer<> {
public:
  explicit SimTauProducer(const edm::ParameterSet&);
  ~SimTauProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  void buildSimTau(SimTauCPLink&,
                   uint8_t,
                   int,
                   const reco::GenParticle&,
                   int,
                   edm::Handle<std::vector<CaloParticle>>,
                   const std::vector<int>&) const;
  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticles_token_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_token_;
  const edm::EDGetTokenT<std::vector<int>> genBarcodes_token_;
};

SimTauProducer::SimTauProducer(const edm::ParameterSet& iConfig)
    : caloParticles_token_(consumes<std::vector<CaloParticle>>(iConfig.getParameter<edm::InputTag>("caloParticles"))),
      genParticles_token_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
      genBarcodes_token_(consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("genBarcodes"))) {
  produces<std::vector<SimTauCPLink>>();
}

void SimTauProducer::buildSimTau(SimTauCPLink& t,
                                 uint8_t generation,
                                 int resonance_idx,
                                 const reco::GenParticle& gen_particle,
                                 int gen_particle_key,
                                 edm::Handle<std::vector<CaloParticle>> calo_particle_h,
                                 const std::vector<int>& gen_particle_barcodes) const {
  const auto& caloPartVec = *calo_particle_h;
  auto& daughters = gen_particle.daughterRefVector();
  bool is_leaf = (daughters.empty());

  if (is_leaf) {
    auto const& gen_particle_barcode = gen_particle_barcodes[gen_particle_key];
    LogDebug("SimTauProducer")
        .format(" TO BE SAVED {}, key {}, barcode {}", resonance_idx, gen_particle_key, gen_particle_barcode);
    auto const& found_in_caloparticles = std::find_if(caloPartVec.begin(), caloPartVec.end(), [&](const auto& p) {
      return p.g4Tracks()[0].genpartIndex() == gen_particle_barcode;
    });
    if (found_in_caloparticles != caloPartVec.end()) {
      auto calo_particle_idx = (found_in_caloparticles - caloPartVec.begin());
      t.calo_particle_leaves.push_back(CaloParticleRef(calo_particle_h, calo_particle_idx));
      t.leaves.push_back(
          {gen_particle.pdgId(), resonance_idx, (int)t.calo_particle_leaves.size() - 1, gen_particle_key});
      LogDebug("SimTauProducer").format(" CP {} {}", calo_particle_idx, caloPartVec[calo_particle_idx].pdgId());
    } else {
      t.leaves.push_back({gen_particle.pdgId(), resonance_idx, -1, gen_particle_key});
    }
    return;
  } else if (generation != 0) {
    t.resonances.push_back({gen_particle.pdgId(), resonance_idx});
    resonance_idx = t.resonances.size() - 1;
    LogDebug("SimTauProducer").format(" RESONANCE/INTERMEDIATE {}", resonance_idx);
  }

  ++generation;
  for (auto daughter = daughters.begin(); daughter != daughters.end(); ++daughter) {
    int gen_particle_key = (*daughter).key();
    LogDebug("SimTauProducer").format(" gen {} {} {}", generation, gen_particle_key, (*daughter)->pdgId());
    buildSimTau(t, generation, resonance_idx, *(*daughter), gen_particle_key, calo_particle_h, gen_particle_barcodes);
  }
}

void SimTauProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  using namespace edm;

  auto caloParticles_h = iEvent.getHandle(caloParticles_token_);
  const auto& genParticles = iEvent.get(genParticles_token_);
  const auto& genBarcodes = iEvent.get(genBarcodes_token_);

  auto tauDecayVec = std::make_unique<std::vector<SimTauCPLink>>();
  for (auto const& g : genParticles) {
    auto const& flags = g.statusFlags();
    if (std::abs(g.pdgId()) == 15 and flags.isPrompt() and flags.isDecayedLeptonHadron()) {
      SimTauCPLink t;
      buildSimTau(t, 0, -1, g, -1, caloParticles_h, genBarcodes);
      t.decayMode = t.buildDecayModes();
#ifdef EDM_ML_DEBUG
      t.dumpFullDecay();
      t.dump();
#endif
      (*tauDecayVec).push_back(t);
    }
  }
  iEvent.put(std::move(tauDecayVec));
}

void SimTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("genBarcodes", edm::InputTag("genParticles"));
  descriptions.add("SimTauProducer", desc);
}

DEFINE_FWK_MODULE(SimTauProducer);
