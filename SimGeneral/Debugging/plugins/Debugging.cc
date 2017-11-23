//
// Original Author:  Marco Rovere
//         Created:  Fri, 10 Nov 2017 14:39:18 GMT
//
//

// system include files
#include <memory>
#include <iostream>
#include <numeric>

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

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>

using namespace boost;
// Graph Definitions
// The graph is meant to represent the full decay chain.
// The parent-child relationship is the natural one, following "time".
// Each edge has a property (edge_weight_t) that holds a const pointer to the SimTrack that connects the 2 vertices of the edge.
// Each vertex has a property (vertex_name_t) that holds a const pointer to the SimTrack that originated that vertex.
// Stable particles are recovered/added in a second iterations and are linked to ghost vertices with a reasonably high offset(1E6).
struct EdgeProperty {
  EdgeProperty(const SimTrack* t, int h, float e) : simTrack(t), simHits(h), energy(e) {};
  const SimTrack* simTrack;
  int simHits;
  float energy;
};

typedef property<edge_weight_t, const SimTrack*> EdgeParticleProperty;
typedef property<edge_weight_t, EdgeProperty> EdgeParticleClustersProperty;
typedef property<vertex_name_t, const SimTrack*> VertexMotherParticleProperty;
typedef adjacency_list< listS, vecS, directedS,
                        VertexMotherParticleProperty, EdgeParticleClustersProperty> DecayChain;

namespace {
  template < typename Edge, typename Graph>
    void print_edge( Edge &e, const Graph & g) {
      auto const edge_property = get(edge_weight, g, e);
      std::cout << "Examining edges " << e
        << " --> particle " << edge_property.simTrack->type()
        << "(" << edge_property.simTrack->trackId() << ")"
        << " with SimClusters: " << edge_property.simHits
        << " and total Energy: " << edge_property.energy << std::endl;
    }
  template < typename Vertex, typename Graph >
    void print_vertex (Vertex &u, const Graph & g) {
      auto const sim_track = get(vertex_name, g, u);
      std::cout << "At " << u;
      // The Mother of all vertices has **no** SimTrack associated.
      if (sim_track)
        std::cout << "[" << sim_track->type() << "]"
                  << "(" << sim_track->trackId() << ")";
      std::cout << std::endl;
    }
  class Custom_dfs_visitor : public boost::default_dfs_visitor
  {
    public:
      template < typename Vertex, typename Graph >
        void examine_vertex(Vertex u, const Graph & g) const {
          print_vertex(u, g);
        }
      template < typename Edge, typename Graph >
        void examine_edge(Edge e, const Graph& g) const {
          print_edge(e, g);
        }
  };
  class Custom_bfs_visitor : public boost::default_bfs_visitor
  {
    public:
      template < typename Vertex, typename Graph >
        void examine_vertex(Vertex u, const Graph & g) const {
          print_vertex(u, g);
        }
      template < typename Edge, typename Graph >
        void examine_edge(Edge e, const Graph& g) const {
          print_edge(e, g);
        }
  };
}

//
// class declaration
//

class Debugging : public edm::one::EDAnalyzer<>  {
   public:
      explicit Debugging(const edm::ParameterSet&);
      ~Debugging();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      void fillSimHits(std::vector<std::pair<DetId, const PCaloHit*> >&,
          std::map<int, std::map<int, float> > &,
          std::map<int, float> &,
          const edm::Event& , const edm::EventSetup &);
      edm::InputTag simTracks_;
      edm::InputTag genParticles_;
      edm::InputTag simVertices_;
      edm::InputTag trackingParticles_;
      edm::InputTag caloParticles_;
      std::vector<edm::InputTag> collectionTags_;
      edm::EDGetTokenT<std::vector<SimTrack> > simTracksToken_;
      edm::EDGetTokenT<std::vector<reco::GenParticle> > genParticlesToken_;
      edm::EDGetTokenT<std::vector<SimVertex> > simVerticesToken_;
      edm::EDGetTokenT<std::vector<TrackingParticle> > trackingParticlesToken_;
      edm::EDGetTokenT<std::vector<CaloParticle> > caloParticlesToken_;
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
Debugging::Debugging(const edm::ParameterSet& iConfig)
  : simTracks_(iConfig.getParameter<edm::InputTag>("simTracks")),
  genParticles_(iConfig.getParameter<edm::InputTag>("genParticles")),
  simVertices_(iConfig.getParameter<edm::InputTag>("simVertices")),
  trackingParticles_(iConfig.getParameter<edm::InputTag>("trackingParticles")),
  caloParticles_(iConfig.getParameter<edm::InputTag>("caloParticles")),
  collectionTags_(iConfig.getParameter<std::vector<edm::InputTag> >("collectionTags"))
{
  edm::ConsumesCollector&& iC = consumesCollector();
  simTracksToken_ = iC.consumes<std::vector<SimTrack> >(simTracks_);
  genParticlesToken_ = iC.consumes<std::vector<reco::GenParticle> > (genParticles_);
  simVerticesToken_ = iC.consumes<std::vector<SimVertex> >(simVertices_);
  trackingParticlesToken_ = iC.consumes<std::vector<TrackingParticle> >(trackingParticles_);
  caloParticlesToken_ = iC.consumes<std::vector<CaloParticle> >(caloParticles_);
  for( const auto& collectionTag : collectionTags_ ) {
    collectionTagsToken_.push_back(iC.consumes<std::vector<PCaloHit> >(collectionTag));
  }
}

Debugging::~Debugging()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
Debugging::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<std::vector<SimTrack> > simTracksH;
  edm::Handle<std::vector<reco::GenParticle> > genParticlesH;
  edm::Handle<std::vector<SimVertex> > simVerticesH;
  edm::Handle<std::vector<TrackingParticle> > trackingParticlesH;
  edm::Handle<std::vector<CaloParticle> > caloParticlesH;

  iEvent.getByToken(simTracksToken_, simTracksH);
  auto const & tracks = *simTracksH.product();

  iEvent.getByToken(genParticlesToken_, genParticlesH);
  auto const & genParticles = *genParticlesH.product();

  iEvent.getByToken(simVerticesToken_, simVerticesH);
  auto const & vertices = *simVerticesH.product();

  iEvent.getByToken(trackingParticlesToken_, trackingParticlesH);
  auto const & trackingpart = *trackingParticlesH.product();

  iEvent.getByToken(caloParticlesToken_, caloParticlesH);
  auto const & calopart = *caloParticlesH.product();

  // Let's first fill in hits information
  std::vector<std::pair<DetId,const PCaloHit*> > simHitPointers;
  std::map<int, std::map<int, float> > simTrackDetIdEnergyMap;
  std::map<int, float> detIdToTotalSimEnergy;
  fillSimHits(simHitPointers, simTrackDetIdEnergyMap, detIdToTotalSimEnergy, iEvent, iSetup);

  int idx = 0;

  std::map<int, int> trackid_to_track_index;
  std::cout << "Printing SimTracks information" << std::endl;
  std::cout << "IDX\tTrackId\tPDGID\tMOMENTUM(x,y,z,E)\tVertexIdx\tGenPartIdx" << std::endl;
  for (auto const & t : tracks) {
    std::cout << idx << "\t" << t.trackId() << "\t" << t << std::endl;
    trackid_to_track_index[t.trackId()] = idx;
    idx++;
  }

  std::cout << "Printing GenParticles information" << std::endl;
  std::cout << "IDX\tPDGID\tMOMENTUM(x,y,z)\tVertex(x,y,z)" << std::endl;
  idx = 0;
  for (auto const & gp : genParticles)
    std::cout << idx++
      << "\t" << gp.pdgId()
      << "\t" << gp.momentum()
      << "\t" << gp.vertex() << std::endl;

  std::cout << "Printing SimVertex information" << std::endl;
  std::cout << "IDX\tPOSITION(x,y,z)\tPARENT_INDEX\tVERTEX_ID" << std::endl;
  DecayChain decay;
  idx = 0;
  // Build the main decay graph and assign the SimTrack to each edge. The graph
  // built here will only contain the particles that have a decay vertex
  // associated to them. In order to recover also the particles that will not
  // decay, we need to keep track of the SimTrack used here and add,
  // a-posteriori, the ones not used, associating a ghost vertex (starting from
  // 1E6), in order to build the edge and identify them immediately as
  // stable (i.e. not decayed).
  std::vector<bool> used_sim_tracks(tracks.size(), false);
  for (auto const & v: vertices) {
    std::cout << idx++ << "\t" << v << std::endl;
    if (v.parentIndex() != -1) {
      add_edge(tracks.at(trackid_to_track_index[v.parentIndex()]).vertIndex(),
          v.vertexId(),
          EdgeProperty(&tracks.at(trackid_to_track_index[v.parentIndex()]),
            simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].size(),
            std::accumulate(simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].begin(),
              simTrackDetIdEnergyMap[trackid_to_track_index[v.parentIndex()]].end(), 0.,
              [&](float partial, std::pair<int, float> current) {
              return partial + current.second/detIdToTotalSimEnergy[current.first]; })),
          decay);
      used_sim_tracks[trackid_to_track_index[v.parentIndex()]] = true;
    }
  }
  // Now recover the particles that did not decay.
  int offset = 1000000;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!used_sim_tracks[i])
      add_edge(tracks.at(i).vertIndex(), offset++,
               EdgeProperty(&tracks.at(i),
                 simTrackDetIdEnergyMap[tracks.at(i).trackId()].size(),
                 std::accumulate(simTrackDetIdEnergyMap[tracks.at(i).trackId()].begin(),
                                 simTrackDetIdEnergyMap[tracks.at(i).trackId()].end(), 0.,
                                 [&](float partial, std::pair<int, float> current) {
                                    return partial + current.second/detIdToTotalSimEnergy[current.first];
                                 })), decay);
  }
  // Now assign the motherParticle property to each vertex
  auto const & vertexMothersProp = get(vertex_name, decay);
  for (auto const & v: vertices) {
    if (v.parentIndex() != -1) {
      put(vertexMothersProp, v.vertexId(), &tracks.at(trackid_to_track_index[v.parentIndex()]));
    }
  }

#define DFS
#ifdef  DFS
  Custom_dfs_visitor vis;
  depth_first_search(decay, visitor(vis));
#else
  Custom_bfs_visitor vis;
  breadth_first_search(decay, 0, visitor(vis));
#endif

  std::cout << "Printing TrackingParticles information" << std::endl;
  idx = 0;
  for (auto const & tp : trackingpart)
    std::cout << idx++ << "\t" << tp << std::endl;

  std::cout << "Printing CaloParticles information" << std::endl;
  idx = 0;
  for (auto const & cp : calopart)
    std::cout << idx++ << " |Eta|: " << std::abs(cp.momentum().eta())
              << "\tEnergy: " << cp.energy() << "\t" << cp << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void
Debugging::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
Debugging::endJob()
{
}

void Debugging::fillSimHits(
    std::vector<std::pair<DetId, const PCaloHit*> >& returnValue,
    std::map<int, std::map<int, float> > & simTrackDetIdEnergyMap,
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

  eegeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward,HGCEE));
  fhgeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward,HGCHEF));
  bhgeom = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal,HcalEndcap));

  hgtopo[0] = &(eegeom->topology());
  hgtopo[1] = &(fhgeom->topology());

  for( unsigned i = 0; i < 2; ++i ) {
    hgddd[i] = &(hgtopo[i]->dddConstants());
  }

  hcddd = bhgeom->topology().dddConstants();

  // loop over the collections
  int token = 0;
  for( const auto& collectionTag : collectionTags_ ) {
    edm::Handle< std::vector<PCaloHit> > hSimHits;
    std::cout << collectionTag.instance() << std::endl;
    const bool isHcal = ( collectionTag.instance().find("HcalHits") != std::string::npos );
    iEvent.getByToken(collectionTagsToken_[token++], hSimHits);
    for( const auto& simHit : *hSimHits ) {
      DetId id(0);
      const uint32_t simId = simHit.id();
      if( isHcal ) {
        HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd);
        if(hid.subdet()==HcalEndcap) id = hid;
      } else {
        int subdet, layer, cell, sec, subsec, zp;
        HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
        const HGCalDDDConstants* ddd = hgddd[subdet-3];
        std::pair<int,int> recoLayerCell = ddd->simToReco(cell,layer,sec,
            hgtopo[subdet-3]->detectorType());
        cell  = recoLayerCell.first;
        layer = recoLayerCell.second;
        // skip simhits with bad barcodes or non-existant layers
        if( layer == -1  || simHit.geantTrackId() == 0 ) continue;
        id = HGCalDetId((ForwardSubdetector)subdet,zp,layer,subsec,sec,cell);
      }

      if( DetId(0) == id ) continue;

      returnValue.emplace_back(id, &simHit);
      if ( simTrackDetIdEnergyMap.count(simHit.geantTrackId())
          && simTrackDetIdEnergyMap[simHit.geantTrackId()].count(id.rawId()))
        simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();
      else
        simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] = simHit.energy();
      if( detIdToTotalSimEnergy.count(id.rawId()) ) detIdToTotalSimEnergy[id.rawId()] += simHit.energy();
      else detIdToTotalSimEnergy[id.rawId()] = simHit.energy();
    }
  } // end of loop over InputTags
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Debugging::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::vector<edm::InputTag> >("collectionTags", {edm::InputTag("g4SimHits","HGCHitsEE"),
                                                           edm::InputTag("g4SimHits","HGCHitsHEfront"),
                                                           edm::InputTag("g4SimHits","HcalHits")});
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Debugging);
