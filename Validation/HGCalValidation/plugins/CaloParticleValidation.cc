// -*- C++ -*-
//
// Class:      CaloParticleValidation
// Original Author:  Marco Rovere
// Created:  Thu, 18 Jan 2018 15:54:55 GMT
//
//

#include <string>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

//
// class declaration
//

struct Histogram_CaloParticleSingle {
  ConcurrentMonitorElement eta_;
  ConcurrentMonitorElement pt_;
  ConcurrentMonitorElement energy_;
  ConcurrentMonitorElement nSimClusters_;
  ConcurrentMonitorElement nHitInSimClusters_;
  ConcurrentMonitorElement selfEnergy_; // this is the sum of the energy associated to all recHits linked to all SimClusters
  ConcurrentMonitorElement energyDifference_; // This contains (energy-selfEnergy)/energy
  ConcurrentMonitorElement eta_Zorigin_map_;
};

using Histograms_CaloParticleValidation = std::unordered_map<int, Histogram_CaloParticleSingle>;

class CaloParticleValidation : public DQMGlobalEDAnalyzer<Histograms_CaloParticleValidation> {
   public:
      explicit CaloParticleValidation(const edm::ParameterSet&);
      ~CaloParticleValidation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void bookHistograms(DQMStore::ConcurrentBooker &,
                                  edm::Run const&,
                                  edm::EventSetup const&,
                                  Histograms_CaloParticleValidation&) const override;

      virtual void dqmAnalyze(edm::Event const&,
                              edm::EventSetup const&,
                              Histograms_CaloParticleValidation const&) const override;

      // ----------member data ---------------------------
      std::string folder_;
      std::vector<int> particles_to_monitor_;

      edm::EDGetTokenT<std::vector<SimVertex> > simVertices_;
      edm::EDGetTokenT<std::vector<CaloParticle> > caloParticles_;
      edm::EDGetTokenT<HGCRecHitCollection> recHitsEE_;
      edm::EDGetTokenT<HGCRecHitCollection> recHitsFH_;
      edm::EDGetTokenT<HGCRecHitCollection> recHitsBH_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloParticleValidation::CaloParticleValidation(const edm::ParameterSet& iConfig)
  : folder_(iConfig.getParameter<std::string>("folder")),
  particles_to_monitor_(iConfig.getParameter<std::vector<int> >("particles_to_monitor")),
  simVertices_(consumes<std::vector<SimVertex>>(iConfig.getParameter<edm::InputTag>("simVertices"))),
  caloParticles_(consumes<std::vector<CaloParticle> >(iConfig.getParameter<edm::InputTag>("caloParticles"))),
  recHitsEE_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEE"))),
  recHitsFH_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsFH"))),
  recHitsBH_(consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsBH")))
{
   //now do what ever initialization is needed
}


CaloParticleValidation::~CaloParticleValidation()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------

void
CaloParticleValidation::dqmAnalyze(edm::Event const& iEvent, edm::EventSetup const& iSetup,
                      Histograms_CaloParticleValidation const & histos) const
{
  using namespace edm;

  Handle<HGCRecHitCollection> recHitHandleEE;
  Handle<HGCRecHitCollection> recHitHandleFH;
  Handle<HGCRecHitCollection> recHitHandleBH;
  // make a map detid-rechit

  iEvent.getByToken(recHitsEE_, recHitHandleEE);
  iEvent.getByToken(recHitsFH_, recHitHandleFH);
  iEvent.getByToken(recHitsBH_, recHitHandleBH);
  const auto& rechitsEE = *recHitHandleEE;
  const auto& rechitsFH = *recHitHandleFH;
  const auto& rechitsBH = *recHitHandleBH;
  std::map<DetId, const HGCRecHit*> hitmap;
  for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
    hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
  }
  for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
    hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
  }
  for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
    hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
  }

  Handle<std::vector<SimVertex>> simVerticesHandle;
  iEvent.getByToken(simVertices_, simVerticesHandle);
  std::vector<SimVertex> const & simVertices = *simVerticesHandle;

  Handle<std::vector<CaloParticle> > caloParticleHandle;
  iEvent.getByToken(caloParticles_, caloParticleHandle);
  std::vector<CaloParticle> const & caloParticles = *caloParticleHandle;

  for (auto const caloParticle : caloParticles) {
    int id = caloParticle.pdgId();
    if (histos.count(id)) {
      histos.at(id).eta_.fill(caloParticle.eta());
      histos.at(id).pt_.fill(caloParticle.pt());
      histos.at(id).energy_.fill(caloParticle.energy());
      histos.at(id).nSimClusters_.fill(caloParticle.simClusters().size());
      // Find the corresponding vertex.
      histos.at(id).eta_Zorigin_map_.fill(
          simVertices.at(caloParticle.g4Tracks()[0].vertIndex()).position().z(), caloParticle.eta());
      int simHits = 0;
      float energy = 0.;
      for (auto const sc : caloParticle.simClusters()) {
        simHits += sc->hits_and_fractions().size();
        for (auto const h_and_f : sc->hits_and_fractions()) {
          if (hitmap.count(h_and_f.first))
            energy += hitmap[h_and_f.first]->energy() * h_and_f.second;
        }
      }
      histos.at(id).nHitInSimClusters_.fill((float)simHits);
      histos.at(id).selfEnergy_.fill(energy);
      histos.at(id).energyDifference_.fill(1.- energy/caloParticle.energy());
    }
  }
}


void
CaloParticleValidation::bookHistograms(DQMStore::ConcurrentBooker & ibook,
                          edm::Run const & run,
                          edm::EventSetup const & iSetup,
                          Histograms_CaloParticleValidation & histos) const
{
  for (auto const particle : particles_to_monitor_) {
    ibook.setCurrentFolder(folder_ + std::to_string(particle));
    histos[particle].eta_ = ibook.book1D("Eta", "Eta", 80, -4., 4.);
    histos[particle].energy_ = ibook.book1D("Energy", "Energy", 250, 0., 500.);
    histos[particle].pt_ = ibook.book1D("Pt", "Pt", 100, 0., 100.);
    histos[particle].nSimClusters_ = ibook.book1D("NSimClusters", "NSimClusters", 100, 0., 100.);
    histos[particle].nHitInSimClusters_ = ibook.book1D("NHitInSimClusters", "NHitInSimClusters", 100, 0., 100.);
    histos[particle].selfEnergy_ = ibook.book1D("SelfEnergy", "SelfEnergy", 250, 0., 500.);
    histos[particle].energyDifference_ = ibook.book1D("EnergyDifference", "(Energy-SelfEnergy)/Energy", 300, -5., 1.);
    histos[particle].eta_Zorigin_map_ = ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", 80, -4., 4., 1100, -550., 550.);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CaloParticleValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/"); // Please keep the trailing '/'
  desc.add<std::vector<int> > ("particles_to_monitor", {11, -11, 13, 22, 111, 211, -211});
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("recHitsEE", edm::InputTag("HGCalRecHit","HGCEERecHits"));
  desc.add<edm::InputTag>("recHitsFH", edm::InputTag("HGCalRecHit","HGCHEFRecHits"));
  desc.add<edm::InputTag>("recHitsBH", edm::InputTag("HGCalRecHit","HGCHEBRecHits"));
  descriptions.add("caloparticlevalidation", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloParticleValidation);
