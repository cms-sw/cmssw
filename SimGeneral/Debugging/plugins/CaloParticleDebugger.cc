//
// Original Author:  Marco Rovere
//         Created:  Fri, 10 Nov 2017 14:39:18 GMT
//
//
//
// system include files
#include <memory>
#include <iostream>
#include <numeric>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"


//
// class declaration
//

class CaloParticleDebugger : public edm::one::EDAnalyzer<>  {
   public:
      explicit CaloParticleDebugger(const edm::ParameterSet&);
      ~CaloParticleDebugger() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void beginJob() override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override;
      void fillSimHits(std::map<int, float> &,
          const edm::Event& , const edm::EventSetup &);
      edm::InputTag simTracks_;
      edm::InputTag genParticles_;
      edm::InputTag simVertices_;
      edm::InputTag trackingParticles_;
      edm::InputTag caloParticles_;
      edm::InputTag simClusters_;
      std::vector<edm::InputTag> collectionTags_;
      edm::EDGetTokenT<std::vector<SimTrack> > simTracksToken_;
      edm::EDGetTokenT<std::vector<reco::GenParticle> > genParticlesToken_;
      edm::EDGetTokenT<std::vector<SimVertex> > simVerticesToken_;
      edm::EDGetTokenT<std::vector<TrackingParticle> > trackingParticlesToken_;
      edm::EDGetTokenT<std::vector<CaloParticle> > caloParticlesToken_;
      edm::EDGetTokenT<std::vector<SimCluster>> simClustersToken_;
      std::vector<edm::EDGetTokenT<std::vector<PCaloHit> > > collectionTagsToken_;
      // ----------member data ---------------------------
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
CaloParticleDebugger::CaloParticleDebugger(const edm::ParameterSet& iConfig)
  : simTracks_(iConfig.getParameter<edm::InputTag>("simTracks")),
  genParticles_(iConfig.getParameter<edm::InputTag>("genParticles")),
  simVertices_(iConfig.getParameter<edm::InputTag>("simVertices")),
  trackingParticles_(iConfig.getParameter<edm::InputTag>("trackingParticles")),
  caloParticles_(iConfig.getParameter<edm::InputTag>("caloParticles")),
  simClusters_(iConfig.getParameter<edm::InputTag>("simClusters")),
  collectionTags_(iConfig.getParameter<std::vector<edm::InputTag> >("collectionTags")) {
  edm::ConsumesCollector&& iC = consumesCollector();
  simTracksToken_ = iC.consumes<std::vector<SimTrack> >(simTracks_);
  genParticlesToken_ = iC.consumes<std::vector<reco::GenParticle> > (genParticles_);
  simVerticesToken_ = iC.consumes<std::vector<SimVertex> >(simVertices_);
  trackingParticlesToken_ = iC.consumes<std::vector<TrackingParticle> >(trackingParticles_);
  caloParticlesToken_ = iC.consumes<std::vector<CaloParticle> >(caloParticles_);
  simClustersToken_ = iC.consumes<std::vector<SimCluster> >(simClusters_);
  for (auto const & collectionTag : collectionTags_) {
    collectionTagsToken_.push_back(iC.consumes<std::vector<PCaloHit> >(collectionTag));
  }
}

CaloParticleDebugger::~CaloParticleDebugger() {}


//
// member functions
//

// ------------ method called for each event  ------------
void
CaloParticleDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using std::begin;
  using std::end;
  using std::sort;
  using std::iota;

  edm::Handle<std::vector<SimTrack> > simTracksH;
  edm::Handle<std::vector<reco::GenParticle> > genParticlesH;
  edm::Handle<std::vector<SimVertex> > simVerticesH;
  edm::Handle<std::vector<TrackingParticle> > trackingParticlesH;
  edm::Handle<std::vector<CaloParticle> > caloParticlesH;
  edm::Handle<std::vector<SimCluster> > simClustersH;

  iEvent.getByToken(simTracksToken_, simTracksH);
  auto const & tracks = *simTracksH.product();
  std::vector<int> sorted_tracks_idx(tracks.size());
  iota(begin(sorted_tracks_idx), end(sorted_tracks_idx), 0);
  sort(begin(sorted_tracks_idx),
       end(sorted_tracks_idx),
       [&tracks] (int i, int j) {
        return tracks[i].momentum().eta() < tracks[j].momentum().eta();
        });

  iEvent.getByToken(genParticlesToken_, genParticlesH);
  auto const & genParticles = *genParticlesH.product();
  std::vector<int> sorted_genParticles_idx(genParticles.size());
  iota(begin(sorted_genParticles_idx), end(sorted_genParticles_idx), 0);
  sort(begin(sorted_genParticles_idx),
       end(sorted_genParticles_idx), [&genParticles](int i, int j) {
       return genParticles[i].momentum().eta() < genParticles[j].momentum().eta();});

  iEvent.getByToken(simVerticesToken_, simVerticesH);
  auto const & vertices = *simVerticesH.product();
  std::vector<int> sorted_vertices_idx(vertices.size());
  iota(begin(sorted_vertices_idx), end(sorted_vertices_idx), 0);
  sort(begin(sorted_vertices_idx),
       end(sorted_vertices_idx), [&vertices](int i, int j){
        return vertices[i].vertexId() < vertices[j].vertexId();
      });

  iEvent.getByToken(trackingParticlesToken_, trackingParticlesH);
  auto const & trackingpart = *trackingParticlesH.product();
  std::vector<int> sorted_tp_idx(trackingpart.size());
  iota(begin(sorted_tp_idx), end(sorted_tp_idx), 0);
  sort(begin(sorted_tp_idx),
       end(sorted_tp_idx), [&trackingpart] (int i, int j){
        return trackingpart[i].eta() < trackingpart[j].eta();
       });

  iEvent.getByToken(caloParticlesToken_, caloParticlesH);
  auto const & calopart = *caloParticlesH.product();
  std::vector<int> sorted_cp_idx(calopart.size());
  iota(begin(sorted_cp_idx),
       end(sorted_cp_idx), 0);
  sort(begin(sorted_cp_idx),
       end(sorted_cp_idx), [&calopart](int i, int j){
       return calopart[i].eta() < calopart[j].eta();});

  iEvent.getByToken(simClustersToken_, simClustersH);
  auto const & simclusters = *simClustersH.product();
  std::vector<int> sorted_simcl_idx(simclusters.size());
  iota(begin(sorted_simcl_idx),
       end(sorted_simcl_idx), 0);
  sort(begin(sorted_simcl_idx),
       end(sorted_simcl_idx), [&simclusters](int i, int j){
       return simclusters[i].eta() < simclusters[j].eta();});

  // Let's first fill in hits information
  std::map<int, float> detIdToTotalSimEnergy;
  fillSimHits(detIdToTotalSimEnergy, iEvent, iSetup);

  int idx = 0;

  std::map<int, int> trackid_to_track_index;
  std::cout << "Printing SimTracks information" << std::endl;
  std::cout << "IDX\tTrackId\tPDGID\tMOMENTUM(x,y,z,E)\tVertexIdx\tGenPartIdx" << std::endl;
  for (auto i : sorted_tracks_idx) {
    auto const & t = tracks[i];
    std::cout << idx << "\t" << t.trackId() << "\t" << t << std::endl;
    trackid_to_track_index[t.trackId()] = idx;
    idx++;
  }

  std::cout << "Printing GenParticles information" << std::endl;
  std::cout << "IDX\tPDGID\tMOMENTUM(x,y,z)\tVertex(x,y,z)" << std::endl;
  for (auto i : sorted_genParticles_idx) {
    auto const & gp = genParticles[i];
    std::cout << i
      << "\t" << gp.pdgId()
      << "\t" << gp.momentum()
      << "\t" << gp.vertex() << std::endl;
  }

  std::cout << "Printing SimVertex information" << std::endl;
  std::cout << "IDX\tPOSITION(x,y,z)\tPARENT_INDEX\tVERTEX_ID" << std::endl;
  for (auto i : sorted_vertices_idx) {
    auto const & v = vertices[i];
      std::cout << i << "\t" << v << std::endl;
  }
  std::cout << "Printing TrackingParticles information" << std::endl;
  for (auto i : sorted_tp_idx) {
    auto const & tp = trackingpart[i];
    std::cout << i << "\t" << tp << std::endl;
  }

  std::cout << "Printing CaloParticles information" << std::endl;
  idx = 0;
  for (auto i : sorted_cp_idx) {
    auto const & cp = calopart[i];
    std::cout << "\n\n" << idx++ << " |Eta|: " << std::abs(cp.momentum().eta())
              << "\tType: " << cp.pdgId()
              << "\tEnergy: " << cp.energy()
              << "\tIdx: " << cp.g4Tracks()[0].trackId() << std::endl; // << cp << std::endl;
    double total_sim_energy = 0.;
    double total_cp_energy = 0.;
    std::cout << "--> Overall simclusters's size: " << cp.simClusters().size() << std::endl;
    // All the next mess just to print the simClusters ordered
    auto const & simcs = cp.simClusters();
    std::vector<int> sorted_sc_idx(simcs.size());
    iota(begin(sorted_sc_idx), end(sorted_sc_idx), 0);
    sort(begin(sorted_sc_idx),
        end(sorted_sc_idx),
        [&simcs] (int i, int j) {
        return simcs[i]->momentum().eta() < simcs[j]->momentum().eta();
        });
    for (auto i : sorted_sc_idx) {
      std::cout <<  *(simcs[i]);
    }

    for (auto const & sc : cp.simClusters()) {
      for (auto const & cl : sc->hits_and_fractions()) {
        total_sim_energy += detIdToTotalSimEnergy[cl.first]*cl.second;
        total_cp_energy += cp.energy()*cl.second;
      }
    }
    std::cout << "--> Overall SC energy (sum using sim energies): " << total_sim_energy << std::endl;
    std::cout << "--> Overall SC energy (sum using CaloP energies): " << total_cp_energy << std::endl;
  }

  idx = 0;
  std::cout << "Printing SimClusters information" << std::endl;
  for (auto i : sorted_simcl_idx) {
    auto const & simcl = simclusters[i];
    std::cout << "\n\n" << idx++ << " |Eta|: " << std::abs(simcl.momentum().eta())
              << "\tType: " << simcl.pdgId()
              << "\tEnergy: " << simcl.energy()
              << "\tKey: " << i << std::endl; // << simcl << std::endl;
    double total_sim_energy = 0.;
    std::cout << "--> Overall simclusters's size: " << simcl.numberOfRecHits() << std::endl;
    for (auto const & cl : simcl.hits_and_fractions()) {
      total_sim_energy += detIdToTotalSimEnergy[cl.first]*cl.second;
    }
    std::cout << simcl << std::endl;
    std::cout << "--> Overall SimCluster energy (sum using sim energies): " << total_sim_energy << std::endl;
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
CaloParticleDebugger::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void
CaloParticleDebugger::endJob() {}

void CaloParticleDebugger::fillSimHits(
    std::map<int, float> & detIdToTotalSimEnergy,
    const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  // Taken needed quantities from the EventSetup
  edm::ESHandle<CaloGeometry> geom;
  iSetup.get<CaloGeometryRecord>().get(geom);
  const HGCalGeometry *eegeom, *fhgeom;
  const HcalGeometry *bhgeom;
  const HGCalDDDConstants* hgddd[2];
  const HGCalTopology*     hgtopo[2];
  const HcalDDDRecConstants* hcddd;

  eegeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
  fhgeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
  bhgeom = static_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalEndcap));

  hgtopo[0] = &(eegeom->topology());
  hgtopo[1] = &(fhgeom->topology());

  for (unsigned i = 0; i < 2; ++i) {
    hgddd[i] = &(hgtopo[i]->dddConstants());
  }

  hcddd = bhgeom->topology().dddConstants();

  // loop over the collections
  int token = 0;
  for (auto const& collectionTag : collectionTags_) {
    edm::Handle< std::vector<PCaloHit> > hSimHits;
    const bool isHcal = ( collectionTag.instance().find("HcalHits") != std::string::npos );
    iEvent.getByToken(collectionTagsToken_[token++], hSimHits);
    for (auto const& simHit : *hSimHits) {
      DetId id(0);
      const uint32_t simId = simHit.id();
      if (isHcal) {
        HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd);
        if (hid.subdet() == HcalEndcap) id = hid;
      } else {
        int subdet, layer, cell, sec, subsec, zp;
        HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
        const HGCalDDDConstants* ddd = hgddd[subdet-3];
        std::pair<int, int> recoLayerCell = ddd->simToReco(cell, layer, sec,
            hgtopo[subdet-3]->detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        // skip simhits with bad barcodes or non-existant layers
        if (layer == -1 || simHit.geantTrackId() == 0) continue;
        id = HGCalDetId((ForwardSubdetector)subdet, zp, layer, subsec, sec, cell);
      }

      if (DetId(0) == id) continue;

      detIdToTotalSimEnergy[id.rawId()] += simHit.energy();
    }
  } // end of loop over InputTags
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CaloParticleDebugger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::vector<edm::InputTag> >("collectionTags",
      { edm::InputTag("g4SimHits", "HGCHitsEE"),
        edm::InputTag("g4SimHits", "HGCHitsHEfront"),
        edm::InputTag("g4SimHits", "HcalHits")});
  descriptions.add("caloParticleDebugger", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(CaloParticleDebugger);
