#include "SimGeneral/CaloAnalysis/plugins/CaloTruthAccumulatorWithGraph.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include <iterator>
#include <numeric>  // for std::accumulate

#define DEBUG true

/* Graph utility functions */

using namespace boost;
namespace {
template <typename Edge, typename Graph, typename Visitor>
void accumulateSimHits_edge(Edge& e, const Graph& g, Visitor* v) {
  auto const edge_property = get(edge_weight, g, e);
  v->total_simHits += edge_property.simHits;
  IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
      << " Examining edges " << e << " --> particle " << edge_property.simTrack->type() << "("
      << edge_property.simTrack->trackId() << ")"
      << " with SimClusters: " << edge_property.simHits
      << " Accumulated SimClusters: " << v->total_simHits << std::endl;
}
template <typename Vertex, typename Graph>
void print_vertex(Vertex& u, const Graph& g) {
  auto const vertex_property = get(vertex_name, g, u);
  IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph") << " At " << u;
  // The Mother of all vertices has **no** SimTrack associated.
  if (vertex_property.simTrack)
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " [" << vertex_property.simTrack->type() << "]"
        << "(" << vertex_property.simTrack->trackId() << ")";
  IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph") << std::endl;
}

// Graphviz output functions will only be generated in DEBUG mode
#if DEBUG
std::string graphviz_vertex(const VertexProperty & v) {
  std::ostringstream oss;
  oss << "{id: " << (v.simTrack ? v.simTrack->trackId() : 0)
    << ", type: " << (v.simTrack ? v.simTrack->type() : 0)
    << ", chits: " << v.cumulative_simHits
    << "}";
  return oss.str();
}

std::string graphviz_edge(const EdgeProperty & e) {
  std::ostringstream oss;
  oss << "[" << (e.simTrack ? e.simTrack->trackId() : 0)
    << "," << (e.simTrack ? e.simTrack->type() : 0)
    << "," << e.simHits
    << "]";
  return oss.str();
}
#endif

class SimHitsAccumulator_dfs_visitor : public boost::default_dfs_visitor {
 public:
  int total_simHits = 0;
  template <typename Edge, typename Graph>
  void examine_edge(Edge e, const Graph& g) {
    accumulateSimHits_edge(e, g, this);
  }
  template <typename Edge, typename Graph>
  void finish_edge(Edge e, const Graph& g) {
    auto const edge_property = get(edge_weight, g, e);
    auto src = source(e, g);
    auto trg = target(e, g);
    auto cumulative =
        edge_property.simHits + get(vertex_name, g, trg).cumulative_simHits +
        (get(vertex_name, g, src).simTrack
             ? get(vertex_name, g, src).cumulative_simHits
             : 0);  // when we hit the root vertex we have to stop adding back its contribution.
    auto const src_vertex_property = get(vertex_name, g, src);
    put(get(vertex_name, const_cast<Graph&>(g)), src,
        VertexProperty(src_vertex_property.simTrack, cumulative));
    put(get(edge_weight, const_cast<Graph&>(g)), e,
        EdgeProperty(edge_property.simTrack, edge_property.simHits, cumulative));
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " Finished edge: " << e << " Track id: " << get(edge_weight, g, e).simTrack->trackId()
        << " has cumulated " << cumulative << " hits" << std::endl;
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " SrcVtx: " << src << "\t" << get(vertex_name, g, src).simTrack << "\t"
        << get(vertex_name, g, src).cumulative_simHits << std::endl;
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " TrgVtx: " << trg << "\t" << get(vertex_name, g, trg).simTrack << "\t"
        << get(vertex_name, g, trg).cumulative_simHits << std::endl;
  }
};

typedef std::function<bool(EdgeProperty&)> Selector;

class CaloParticle_dfs_visitor : public boost::default_dfs_visitor {
 public:
  CaloParticle_dfs_visitor(CaloTruthAccumulatorWithGraph::OutputCollections& output,
                           CaloTruthAccumulatorWithGraph::calo_particles & caloParticles,
                           std::unordered_multimap<Barcode_t, Index_t>& simHitBarcodeToIndex,
                           std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap,
                           Selector selector)
      : output_(output),
        caloParticles_(caloParticles),
        simHitBarcodeToIndex_(simHitBarcodeToIndex),
        simTrackDetIdEnergyMap_(simTrackDetIdEnergyMap),
        selector_(selector) {}
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph& g) {
    // If we reach the vertex 0, it means that we are backtracking with respect
    // to the first generation of stable particles: simply return;
    if (u == 0)
      return;
    print_vertex(u, g);
    auto const vertex_property = get(vertex_name, g, u);
    if (!vertex_property.simTrack) return;
    auto trackIdx = vertex_property.simTrack->trackId();
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " Found " << simHitBarcodeToIndex_.count(trackIdx) << " associated simHits" << std::endl;
    if (simHitBarcodeToIndex_.count(trackIdx)) {
      output_.pSimClusters->emplace_back(*vertex_property.simTrack);
      auto& simcluster = output_.pSimClusters->back();
      std::unordered_map<uint32_t, float> acc_energy;
      for (const auto& hit_and_energy : simTrackDetIdEnergyMap_[trackIdx]) {
        acc_energy[hit_and_energy.first] += hit_and_energy.second;
      }
      for (const auto& hit_and_energy : acc_energy) {
        simcluster.addRecHitAndFraction(hit_and_energy.first, hit_and_energy.second);
      }
    }
  }
  template <typename Edge, typename Graph>
  void examine_edge(Edge e, const Graph &g) {
    auto src = source(e, g);
    if (src == 0) {
      auto edge_property = get(edge_weight, g, e);
      if (selector_(edge_property)) {
        output_.pCaloParticles->emplace_back(*(edge_property.simTrack));
        caloParticles_.sc_start_.push_back(output_.pSimClusters->size());
      }
    }
  }

  template <typename Edge, typename Graph>
  void finish_edge(Edge e, const Graph & g) {
    auto src = source(e, g);
    if (src == 0) {
      auto edge_property = get(edge_weight, g, e);
      if (selector_(edge_property)) {
        caloParticles_.sc_stop_.push_back(output_.pSimClusters->size());
      }
    }
  }
 private:
  CaloTruthAccumulatorWithGraph::OutputCollections& output_;
  CaloTruthAccumulatorWithGraph::calo_particles & caloParticles_;
  std::unordered_multimap<Barcode_t, Index_t>& simHitBarcodeToIndex_;
  std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap_;
  Selector selector_;
};
}

CaloTruthAccumulatorWithGraph::CaloTruthAccumulatorWithGraph(const edm::ParameterSet& config,
                                                             edm::stream::EDProducerBase& mixMod,
                                                             edm::ConsumesCollector& iC)
    : messageCategory_("CaloTruthAccumulatorWithGraph"),
      maximumPreviousBunchCrossing_(
          config.getParameter<unsigned int>("maximumPreviousBunchCrossing")),
      maximumSubsequentBunchCrossing_(
          config.getParameter<unsigned int>("maximumSubsequentBunchCrossing")),
      simTrackLabel_(config.getParameter<edm::InputTag>("simTrackCollection")),
      simVertexLabel_(config.getParameter<edm::InputTag>("simVertexCollection")),
      collectionTags_(),
      genParticleLabel_(config.getParameter<edm::InputTag>("genParticleCollection")),
      hepMCproductLabel_(config.getParameter<edm::InputTag>("HepMCProductLabel")),
      minEnergy_(config.getParameter<double>("MinEnergy")),
      maxPseudoRapidity_(config.getParameter<double>("MaxPseudoRapidity")) {
  barcodeLogicWarningAlready_ = false;

  mixMod.produces<SimClusterCollection>("MergedCaloTruth");
  mixMod.produces<CaloParticleCollection>("MergedCaloTruth");

  iC.consumes<std::vector<SimTrack> >(simTrackLabel_);
  iC.consumes<std::vector<SimVertex> >(simVertexLabel_);
  iC.consumes<std::vector<reco::GenParticle> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(hepMCproductLabel_);

  // Fill the collection tags
  const edm::ParameterSet& simHitCollectionConfig = config.getParameterSet("simHitCollections");
  std::vector<std::string> parameterNames = simHitCollectionConfig.getParameterNames();

  for (const auto& parameterName : parameterNames) {
    std::vector<edm::InputTag> tags =
        simHitCollectionConfig.getParameter<std::vector<edm::InputTag> >(parameterName);
    collectionTags_.insert(collectionTags_.end(), tags.begin(), tags.end());
  }

  for (const auto& collectionTag : collectionTags_) {
    iC.consumes<std::vector<PCaloHit> >(collectionTag);
  }
}

void CaloTruthAccumulatorWithGraph::beginLuminosityBlock(edm::LuminosityBlock const& iLumiBlock,
                                                         const edm::EventSetup& iSetup) {
  edm::ESHandle<CaloGeometry> geom;
  iSetup.get<CaloGeometryRecord>().get(geom);
  const HGCalGeometry *eegeom, *fhgeom;
  const HcalGeometry* bhgeom;

  eegeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
  fhgeom = dynamic_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
  bhgeom = dynamic_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalEndcap));

  hgtopo_[0] = &(eegeom->topology());
  hgtopo_[1] = &(fhgeom->topology());

  for (unsigned i = 0; i < 2; ++i) {
    hgddd_[i] = &(hgtopo_[i]->dddConstants());
  }

  hcddd_ = bhgeom->topology().dddConstants();

  caloStartZ = hgddd_[0]->waferZ(1, false) *
               10.0;  // get the location of the first plane of silicon, put in mm
}

void CaloTruthAccumulatorWithGraph::initializeEvent(edm::Event const& event,
                                                    edm::EventSetup const& setup) {
  output_.pSimClusters.reset(new SimClusterCollection());
  output_.pCaloParticles.reset(new CaloParticleCollection());

  m_detIdToTotalSimEnergy.clear();
}

/// create handle to edm::HepMCProduct here because event.getByLabel with edm::HepMCProduct only
/// works for edm::Event
/// but not for PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call T::value_type
/// and T::iterator
/// (where T is the type of the object one wants to get a handle to) which is only implemented for
/// container-like objects
/// like std::vector but not for edm::HepMCProduct!

void CaloTruthAccumulatorWithGraph::accumulate(edm::Event const& event,
                                               edm::EventSetup const& setup) {
  // Call the templated version that does the same for both signal and pileup events

  edm::Handle<edm::HepMCProduct> hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);

  edm::LogInfo(messageCategory_) << " CaloTruthAccumulatorWithGraph::accumulate (signal)";
  accumulateEvent(event, setup, hepmc);
}

void CaloTruthAccumulatorWithGraph::accumulate(PileUpEventPrincipal const& event,
                                               edm::EventSetup const& setup, edm::StreamID const&) {
  // If this bunch crossing is outside the user configured limit, don't do anything.
  if (event.bunchCrossing() >= -static_cast<int>(maximumPreviousBunchCrossing_) &&
      event.bunchCrossing() <= static_cast<int>(maximumSubsequentBunchCrossing_)) {
    // edm::LogInfo(messageCategory_) << "Analysing pileup event for bunch crossing " <<
    // event.bunchCrossing();

    // simply create empty handle as we do not have a HepMCProduct in PU anyway
    edm::Handle<edm::HepMCProduct> hepmc;
    edm::LogInfo(messageCategory_)
        << " CaloTruthAccumulatorWithGraph::accumulate (pileup) bunchCrossing="
        << event.bunchCrossing();
    accumulateEvent(event, setup, hepmc);
  } else
    edm::LogInfo(messageCategory_)
        << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
}

void CaloTruthAccumulatorWithGraph::finalizeEvent(edm::Event& event, edm::EventSetup const& setup) {
  edm::LogInfo("CaloTruthAccumulatorWithGraph")
      << "Adding " << output_.pSimClusters->size() << " SimParticles and "
      << output_.pCaloParticles->size() << " CaloParticles to the event.";

  // now we need to normalize the hits and energies into hits and fractions
  // (since we have looped over all pileup events)

  for (auto& sc : *(output_.pSimClusters)) {
    auto hitsAndEnergies = sc.hits_and_fractions();
    sc.clearHitsAndFractions();
    for (auto& hAndE : hitsAndEnergies) {
      const float totalenergy = m_detIdToTotalSimEnergy[hAndE.first];
      float fraction = 0.;
      if (totalenergy > 0)
        fraction = hAndE.second / totalenergy;
      else
        edm::LogWarning("CaloTruthAccumulatorWithGraph")
            << "TotalSimEnergy for hit " << hAndE.first
            << " is 0! The fraction for this hit cannot be computed.";
      sc.addRecHitAndFraction(hAndE.first, fraction);
    }
  }

  // save the SimCluster orphan handle so we can fill the calo particles
  auto scHandle = event.put(std::move(output_.pSimClusters), "MergedCaloTruth");

  // now fill the calo particles
  for (unsigned i = 0; i < output_.pCaloParticles->size(); ++i) {
    auto& cp = (*output_.pCaloParticles)[i];
    for (unsigned j = m_caloParticles.sc_start_[i]; j < m_caloParticles.sc_stop_[i]; ++j) {
      edm::Ref<SimClusterCollection> ref(scHandle, j);
      cp.addSimCluster(ref);
    }
  }

  event.put(std::move(output_.pCaloParticles), "MergedCaloTruth");

  calo_particles().swap(m_caloParticles);

  std::unordered_map<Index_t, float>().swap(m_detIdToTotalSimEnergy);
  std::unordered_multimap<Barcode_t, Index_t>().swap(m_simHitBarcodeToIndex);
}

template <class T>
void CaloTruthAccumulatorWithGraph::accumulateEvent(
    const T& event, const edm::EventSetup& setup,
    const edm::Handle<edm::HepMCProduct>& hepMCproduct) {
  //
  // Get the collections
  //

  edm::Handle<std::vector<reco::GenParticle> > hGenParticles;
  edm::Handle<std::vector<int> > hGenParticleIndices;

  event.getByLabel(simTrackLabel_, hSimTracks);
  event.getByLabel(simVertexLabel_, hSimVertices);

  event.getByLabel(genParticleLabel_, hGenParticles);
  event.getByLabel(genParticleLabel_, hGenParticleIndices);

  std::vector<std::pair<DetId, const PCaloHit*> > simHitPointers;
  std::unordered_map<int, std::map<int, float> > simTrackDetIdEnergyMap;
  fillSimHits(simHitPointers, simTrackDetIdEnergyMap, event, setup);

  // Clear maps from previous event fill them for this one
  m_simHitBarcodeToIndex.clear();
  for (unsigned int i = 0; i < simHitPointers.size(); ++i) {
    m_simHitBarcodeToIndex.emplace(simHitPointers[i].second->geantTrackId(), i);
  }

  const auto& tracks = *hSimTracks;
  const auto& vertices = *hSimVertices;
  std::unordered_map<int, int> trackid_to_track_index;
  DecayChain decay;
  int idx = 0;

  IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph") << " TRACKS" << std::endl;
  for (auto const& t : tracks) {
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph")
        << " " << idx << "\t" << t.trackId() << "\t" << t << std::endl;
    trackid_to_track_index[t.trackId()] = idx;
    idx++;
  }

  // Build the main decay graph and assign the SimTrack to each edge. The graph
  // built here will only contain the particles that have a decay vertex
  // associated to them. In order to recover also the particles that will not
  // decay, we need to keep track of the SimTrack used here and add,
  // a-posteriori, the ones not used, associating a ghost vertex (starting from
  // the highest simulated vertex number +1), in order to build the edge and
  // identify them immediately as stable (i.e. not decayed).
  idx = 0;
  std::vector<int> used_sim_tracks(tracks.size(), 0);
  IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph") << " VERTICES" << std::endl;
  for (auto const& v : vertices) {
    IfLogDebug(DEBUG, "CaloTruthAccumulatorWithGraph") << " " << idx++ << "\t" << v << std::endl;
    if (v.parentIndex() != -1) {
      // TODO(rovere): for multi-bremh do not continue the chain but collapse
      // the multi-vertices in 1 unique vertex
      auto trk_idx = trackid_to_track_index[v.parentIndex()];
      auto origin_vtx = tracks[trk_idx].vertIndex();
      if (used_sim_tracks[trk_idx])
        origin_vtx = used_sim_tracks[trk_idx];
      add_edge(origin_vtx, v.vertexId(),
               EdgeProperty(
                   &tracks[trk_idx],
                   simTrackDetIdEnergyMap[v.parentIndex()].size(), 0),
               decay);
      used_sim_tracks[trk_idx] = v.vertexId();
    }
  }
  // Assign the motherParticle property to each vertex
  auto const& vertexMothersProp = get(vertex_name, decay);
  // Now recover the particles that did not decay. Append them with an index
  // bigger than the size of the generated vertices.
  int offset = vertices.size() + 1;
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!used_sim_tracks[i]) {
      add_edge(tracks[i].vertIndex(), offset,
               EdgeProperty(
                   &tracks[i], simTrackDetIdEnergyMap[tracks[i].trackId()].size(), 0),
               decay);
      // The properties for "fake" vertices associated to stable particles have
      // to be set inside this loop, since they do not belong to the vertices
      // collection and would be skipped by that loop (coming next)
      put(vertexMothersProp, offset, VertexProperty(&tracks[i], 0));
      offset++;
    }
  }
  for (auto const& v : vertices) {
    if (v.parentIndex() != -1) {
      put(vertexMothersProp, v.vertexId(),
          VertexProperty(&tracks[trackid_to_track_index[v.parentIndex()]], 0));
    }
  }
  SimHitsAccumulator_dfs_visitor vis;
  depth_first_search(decay, visitor(vis));
  CaloParticle_dfs_visitor caloParticleCreator(output_,
      m_caloParticles,
      m_simHitBarcodeToIndex,
      simTrackDetIdEnergyMap,
      [&](EdgeProperty & edge_property) -> bool {
     // Apply selection on SimTracks in order to promote them to be CaloParticles.
     // The function returns TRUE if the particle satisfies the selection, FALSE otherwise.
     // Therefore the correct logic to select the particle is to ask for TRUE as return value.
      return (edge_property.cumulative_simHits != 0
        and ! edge_property.simTrack->noGenpart()
        and edge_property.simTrack->momentum().E() > minEnergy_
        and std::abs(edge_property.simTrack->momentum().Eta()) < maxPseudoRapidity_);
      });
  depth_first_search(decay, visitor(caloParticleCreator));

#if DEBUG
  boost::write_graphviz(std::cout, decay,
      make_label_writer(
        make_transform_value_property_map(&graphviz_vertex, get(vertex_name, decay))),
      make_label_writer(
        make_transform_value_property_map(&graphviz_edge, get(edge_weight, decay))));
#endif
}

template <class T>
void CaloTruthAccumulatorWithGraph::fillSimHits(
    std::vector<std::pair<DetId, const PCaloHit*> >& returnValue,
    std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap, const T& event,
    const edm::EventSetup& setup) {
  // loop over the collections
  for (const auto& collectionTag : collectionTags_) {
    edm::Handle<std::vector<PCaloHit> > hSimHits;
    const bool isHcal = (collectionTag.instance().find("HcalHits") != std::string::npos);
    event.getByLabel(collectionTag, hSimHits);
    for (const auto& simHit : *hSimHits) {
      DetId id(0);
      const uint32_t simId = simHit.id();
      if (isHcal) {
        HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd_);
        if (hid.subdet() == HcalEndcap) id = hid;
      } else {
        int subdet, layer, cell, sec, subsec, zp;
        HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
        const HGCalDDDConstants* ddd = hgddd_[subdet - 3];
        std::pair<int, int> recoLayerCell =
            ddd->simToReco(cell, layer, sec, hgtopo_[subdet - 3]->detectorType());
        cell = recoLayerCell.first;
        layer = recoLayerCell.second;
        // skip simhits with bad barcodes or non-existant layers
        if (layer == -1 || simHit.geantTrackId() == 0) continue;
        id = HGCalDetId((ForwardSubdetector)subdet, zp, layer, subsec, sec, cell);
      }

      if (DetId(0) == id) continue;

      uint32_t detId = id.rawId();
      returnValue.emplace_back(id, &simHit);
      simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();

      m_detIdToTotalSimEnergy[detId] += simHit.energy();
    }
  }  // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR(CaloTruthAccumulatorWithGraph);
