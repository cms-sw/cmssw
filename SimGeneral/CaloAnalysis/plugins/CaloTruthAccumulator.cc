#define DEBUG false

#if DEBUG
#pragma GCC diagnostic pop
#endif

#include <iterator>
#include <memory>

#include <numeric>  // for std::accumulate
#include <unordered_map>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "SimGeneral/MixingModule/interface/DecayGraph.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

namespace {
  using Index_t = unsigned;
  using Barcode_t = int;
  const std::string messageCategoryGraph_("CaloTruthAccumulatorGraphProducer");
}  // namespace

class CaloTruthAccumulator : public DigiAccumulatorMixMod {
public:
  explicit CaloTruthAccumulator(const edm::ParameterSet &config, edm::ProducesCollector, edm::ConsumesCollector &iC);

private:
  void initializeEvent(const edm::Event &event, const edm::EventSetup &setup) override;
  void accumulate(const edm::Event &event, const edm::EventSetup &setup) override;
  void accumulate(const PileUpEventPrincipal &event, const edm::EventSetup &setup, edm::StreamID const &) override;
  void finalizeEvent(edm::Event &event, const edm::EventSetup &setup) override;

  /** @brief Both forms of accumulate() delegate to this templated method. */
  template <class T>
  void accumulateEvent(const T &event,
                       const edm::EventSetup &setup,
                       const edm::Handle<edm::HepMCProduct> &hepMCproduct);

  /** @brief Fills the supplied vector with pointers to the SimHits, checking
   * for bad modules if required */
  template <class T>
  void fillSimHits(std::vector<std::pair<DetId, const PCaloHit *>> &returnValue,
                   std::unordered_map<int, std::map<int, float>> &simTrackDetIdEnergyMap,
                   const T &event,
                   const edm::EventSetup &setup);

  const std::string messageCategory_;

  std::unordered_map<Index_t, float> m_detIdToTotalSimEnergy;  // keep track of cell normalizations
  std::unordered_multimap<Barcode_t, Index_t> m_simHitBarcodeToIndex;

  /** The maximum bunch crossing BEFORE the signal crossing to create
      TrackinParticles for. Use positive values. If set to zero no
      previous bunches are added and only in-time, signal and after bunches
      (defined by maximumSubsequentBunchCrossing_) are used.
  */
  const unsigned int maximumPreviousBunchCrossing_;
  /** The maximum bunch crossing AFTER the signal crossing to create
      TrackinParticles for. E.g. if set to zero only
      uses the signal and in time pileup (and previous bunches defined by the
      maximumPreviousBunchCrossing_ parameter).
  */
  const unsigned int maximumSubsequentBunchCrossing_;

  const edm::InputTag simTrackLabel_;
  const edm::InputTag simVertexLabel_;
  edm::Handle<std::vector<SimTrack>> hSimTracks;
  edm::Handle<std::vector<SimVertex>> hSimVertices;

  std::vector<edm::InputTag> collectionTags_;
  edm::InputTag genParticleLabel_;
  /// Needed to add HepMC::GenVertex to SimVertex
  edm::InputTag hepMCproductLabel_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESWatcher<CaloGeometryRecord> geomWatcher_;

  const double minEnergy_, maxPseudoRapidity_;
  const bool premixStage1_;

public:
  struct OutputCollections {
    std::unique_ptr<SimClusterCollection> pSimClusters;
    std::unique_ptr<CaloParticleCollection> pCaloParticles;
  };

  struct calo_particles {
    std::vector<uint32_t> sc_start_;
    std::vector<uint32_t> sc_stop_;

    void swap(calo_particles &oth) {
      sc_start_.swap(oth.sc_start_);
      sc_stop_.swap(oth.sc_stop_);
    }

    void clear() {
      sc_start_.clear();
      sc_stop_.clear();
    }
  };

private:
  const HGCalTopology *hgtopo_[3] = {nullptr, nullptr, nullptr};
  const HGCalDDDConstants *hgddd_[3] = {nullptr, nullptr, nullptr};
  const HcalDDDRecConstants *hcddd_ = nullptr;
  OutputCollections output_;
  calo_particles m_caloParticles;
  // geometry type (0 pre-TDR; 1 TDR)
  int geometryType_;
  bool doHGCAL;
};

/* Graph utility functions */

namespace {
  class CaloParticle_dfs_visitor : public boost::default_dfs_visitor {
  public:
    CaloParticle_dfs_visitor(CaloTruthAccumulator::OutputCollections &output,
                             CaloTruthAccumulator::calo_particles &caloParticles,
                             std::unordered_multimap<Barcode_t, Index_t> &simHitBarcodeToIndex,
                             std::unordered_map<int, std::map<int, float>> &simTrackDetIdEnergyMap,
                             Selector selector)
        : output_(output),
          caloParticles_(caloParticles),
          simHitBarcodeToIndex_(simHitBarcodeToIndex),
          simTrackDetIdEnergyMap_(simTrackDetIdEnergyMap),
          selector_(selector) {}
    template <typename Vertex, typename Graph>
    void discover_vertex(Vertex u, const Graph &g) {
      // If we reach the vertex 0, it means that we are backtracking with respect
      // to the first generation of stable particles: simply return;
      //    if (u == 0) return;
      print_vertex(u, g);
      auto const vertex_property = get(vertex_name, g, u);
      if (!vertex_property.simTrack)
        return;
      auto trackIdx = vertex_property.simTrack->trackId();
      IfLogDebug(DEBUG, messageCategoryGraph_)
          << " Found " << simHitBarcodeToIndex_.count(trackIdx) << " associated simHits" << std::endl;
      if (simHitBarcodeToIndex_.count(trackIdx)) {
        output_.pSimClusters->emplace_back(*vertex_property.simTrack);
        auto &simcluster = output_.pSimClusters->back();
        std::unordered_map<uint32_t, float> acc_energy;
        for (auto const &hit_and_energy : simTrackDetIdEnergyMap_[trackIdx]) {
          acc_energy[hit_and_energy.first] += hit_and_energy.second;
        }
        for (auto const &hit_and_energy : acc_energy) {
          simcluster.addRecHitAndFraction(hit_and_energy.first, hit_and_energy.second);
        }
      }
    }
    template <typename Edge, typename Graph>
    void examine_edge(Edge e, const Graph &g) {
      auto src = source(e, g);
      auto vertex_property = get(vertex_name, g, src);
      if (src == 0 or (vertex_property.simTrack == nullptr)) {
        auto edge_property = get(edge_weight, g, e);
        IfLogDebug(DEBUG, messageCategoryGraph_) << "Considering CaloParticle: " << edge_property.simTrack->trackId();
        if (selector_(edge_property)) {
          IfLogDebug(DEBUG, messageCategoryGraph_) << "Adding CaloParticle: " << edge_property.simTrack->trackId();
          output_.pCaloParticles->emplace_back(*(edge_property.simTrack));
          caloParticles_.sc_start_.push_back(output_.pSimClusters->size());
        }
      }
    }

    template <typename Edge, typename Graph>
    void finish_edge(Edge e, const Graph &g) {
      auto src = source(e, g);
      auto vertex_property = get(vertex_name, g, src);
      if (src == 0 or (vertex_property.simTrack == nullptr)) {
        auto edge_property = get(edge_weight, g, e);
        if (selector_(edge_property)) {
          caloParticles_.sc_stop_.push_back(output_.pSimClusters->size());
        }
      }
    }

  private:
    CaloTruthAccumulator::OutputCollections &output_;
    CaloTruthAccumulator::calo_particles &caloParticles_;
    std::unordered_multimap<Barcode_t, Index_t> &simHitBarcodeToIndex_;
    std::unordered_map<int, std::map<int, float>> &simTrackDetIdEnergyMap_;
    Selector selector_;
  };
}  // namespace

CaloTruthAccumulator::CaloTruthAccumulator(const edm::ParameterSet &config,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector &iC)
    : messageCategory_("CaloTruthAccumulator"),
      maximumPreviousBunchCrossing_(config.getParameter<unsigned int>("maximumPreviousBunchCrossing")),
      maximumSubsequentBunchCrossing_(config.getParameter<unsigned int>("maximumSubsequentBunchCrossing")),
      simTrackLabel_(config.getParameter<edm::InputTag>("simTrackCollection")),
      simVertexLabel_(config.getParameter<edm::InputTag>("simVertexCollection")),
      collectionTags_(),
      genParticleLabel_(config.getParameter<edm::InputTag>("genParticleCollection")),
      hepMCproductLabel_(config.getParameter<edm::InputTag>("HepMCProductLabel")),
      geomToken_(iC.esConsumes()),
      minEnergy_(config.getParameter<double>("MinEnergy")),
      maxPseudoRapidity_(config.getParameter<double>("MaxPseudoRapidity")),
      premixStage1_(config.getParameter<bool>("premixStage1")),
      geometryType_(-1),
      doHGCAL(config.getParameter<bool>("doHGCAL")) {
  producesCollector.produces<SimClusterCollection>("MergedCaloTruth");
  producesCollector.produces<CaloParticleCollection>("MergedCaloTruth");
  if (premixStage1_) {
    producesCollector.produces<std::vector<std::pair<unsigned int, float>>>("MergedCaloTruth");
  }

  iC.consumes<std::vector<SimTrack>>(simTrackLabel_);
  iC.consumes<std::vector<SimVertex>>(simVertexLabel_);
  iC.consumes<std::vector<reco::GenParticle>>(genParticleLabel_);
  iC.consumes<std::vector<int>>(genParticleLabel_);
  iC.consumes<std::vector<int>>(hepMCproductLabel_);

  // Fill the collection tags
  const edm::ParameterSet &simHitCollectionConfig = config.getParameterSet("simHitCollections");
  std::vector<std::string> parameterNames = simHitCollectionConfig.getParameterNames();

  for (auto const &parameterName : parameterNames) {
    std::vector<edm::InputTag> tags = simHitCollectionConfig.getParameter<std::vector<edm::InputTag>>(parameterName);
    collectionTags_.insert(collectionTags_.end(), tags.begin(), tags.end());
  }

  for (auto const &collectionTag : collectionTags_) {
    iC.consumes<std::vector<PCaloHit>>(collectionTag);
  }
}

void CaloTruthAccumulator::initializeEvent(edm::Event const &event, edm::EventSetup const &setup) {
  output_.pSimClusters = std::make_unique<SimClusterCollection>();
  output_.pCaloParticles = std::make_unique<CaloParticleCollection>();

  m_detIdToTotalSimEnergy.clear();

  if (geomWatcher_.check(setup)) {
    auto const &geom = setup.getData(geomToken_);
    const HGCalGeometry *eegeom = nullptr, *fhgeom = nullptr, *bhgeomnew = nullptr;
    const HcalGeometry *bhgeom = nullptr;
    bhgeom = static_cast<const HcalGeometry *>(geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));

    if (doHGCAL) {
      eegeom = static_cast<const HGCalGeometry *>(
          geom.getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty));
      // check if it's the new geometry
      if (eegeom) {
        geometryType_ = 1;
        fhgeom = static_cast<const HGCalGeometry *>(
            geom.getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
        bhgeomnew = static_cast<const HGCalGeometry *>(
            geom.getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));
      } else {
        geometryType_ = 0;
        eegeom = static_cast<const HGCalGeometry *>(geom.getSubdetectorGeometry(DetId::Forward, HGCEE));
        fhgeom = static_cast<const HGCalGeometry *>(geom.getSubdetectorGeometry(DetId::Forward, HGCHEF));
        bhgeom = static_cast<const HcalGeometry *>(geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
      }
      hgtopo_[0] = &(eegeom->topology());
      hgtopo_[1] = &(fhgeom->topology());
      if (bhgeomnew)
        hgtopo_[2] = &(bhgeomnew->topology());

      for (unsigned i = 0; i < 3; ++i) {
        if (hgtopo_[i])
          hgddd_[i] = &(hgtopo_[i]->dddConstants());
      }
    }

    if (bhgeom) {
      hcddd_ = bhgeom->topology().dddConstants();
    }
  }
}

/** Create handle to edm::HepMCProduct here because event.getByLabel with
    edm::HepMCProduct only works for edm::Event but not for
    PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call
    T::value_type and T::iterator (where T is the type of the object one wants
    to get a handle to) which is only implemented for container-like objects
    like std::vector but not for edm::HepMCProduct!
*/
void CaloTruthAccumulator::accumulate(edm::Event const &event, edm::EventSetup const &setup) {
  edm::Handle<edm::HepMCProduct> hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);

  edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (signal)";
  accumulateEvent(event, setup, hepmc);
}

void CaloTruthAccumulator::accumulate(PileUpEventPrincipal const &event,
                                      edm::EventSetup const &setup,
                                      edm::StreamID const &) {
  if (event.bunchCrossing() >= -static_cast<int>(maximumPreviousBunchCrossing_) &&
      event.bunchCrossing() <= static_cast<int>(maximumSubsequentBunchCrossing_)) {
    // simply create empty handle as we do not have a HepMCProduct in PU anyway
    edm::Handle<edm::HepMCProduct> hepmc;
    edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (pileup) bunchCrossing="
                                   << event.bunchCrossing();
    accumulateEvent(event, setup, hepmc);
  } else {
    edm::LogInfo(messageCategory_) << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
  }
}

void CaloTruthAccumulator::finalizeEvent(edm::Event &event, edm::EventSetup const &setup) {
  edm::LogInfo(messageCategory_) << "Adding " << output_.pSimClusters->size() << " SimParticles and "
                                 << output_.pCaloParticles->size() << " CaloParticles to the event.";

  // We need to normalize the hits and energies into hits and fractions (since
  // we have looped over all pileup events)
  // For premixing stage1 we keep the energies, they will be normalized to
  // fractions in stage2

  if (premixStage1_) {
    auto totalEnergies = std::make_unique<std::vector<std::pair<unsigned int, float>>>();
    totalEnergies->reserve(m_detIdToTotalSimEnergy.size());
    std::copy(m_detIdToTotalSimEnergy.begin(), m_detIdToTotalSimEnergy.end(), std::back_inserter(*totalEnergies));
    std::sort(totalEnergies->begin(), totalEnergies->end());
    event.put(std::move(totalEnergies), "MergedCaloTruth");
  } else {
    for (auto &sc : *(output_.pSimClusters)) {
      auto hitsAndEnergies = sc.hits_and_fractions();
      sc.clearHitsAndFractions();
      sc.clearHitsEnergy();
      for (auto &hAndE : hitsAndEnergies) {
        const float totalenergy = m_detIdToTotalSimEnergy[hAndE.first];
        float fraction = 0.;
        if (totalenergy > 0)
          fraction = hAndE.second / totalenergy;
        else
          edm::LogWarning(messageCategory_)
              << "TotalSimEnergy for hit " << hAndE.first << " is 0! The fraction for this hit cannot be computed.";
        sc.addRecHitAndFraction(hAndE.first, fraction);
        sc.addHitEnergy(hAndE.second);
      }
    }
  }

  // save the SimCluster orphan handle so we can fill the calo particles
  auto scHandle = event.put(std::move(output_.pSimClusters), "MergedCaloTruth");

  // now fill the calo particles
  for (unsigned i = 0; i < output_.pCaloParticles->size(); ++i) {
    auto &cp = (*output_.pCaloParticles)[i];
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
void CaloTruthAccumulator::accumulateEvent(const T &event,
                                           const edm::EventSetup &setup,
                                           const edm::Handle<edm::HepMCProduct> &hepMCproduct) {
  edm::Handle<std::vector<reco::GenParticle>> hGenParticles;
  edm::Handle<std::vector<int>> hGenParticleIndices;

  event.getByLabel(simTrackLabel_, hSimTracks);
  event.getByLabel(simVertexLabel_, hSimVertices);

  event.getByLabel(genParticleLabel_, hGenParticles);
  event.getByLabel(genParticleLabel_, hGenParticleIndices);

  std::vector<std::pair<DetId, const PCaloHit *>> simHitPointers;
  std::unordered_map<int, std::map<int, float>> simTrackDetIdEnergyMap;
  fillSimHits(simHitPointers, simTrackDetIdEnergyMap, event, setup);

  // Clear maps from previous event fill them for this one
  m_simHitBarcodeToIndex.clear();
  for (unsigned int i = 0; i < simHitPointers.size(); ++i) {
    m_simHitBarcodeToIndex.emplace(simHitPointers[i].second->geantTrackId(), i);
  }

  auto const &tracks = *hSimTracks;
  auto const &vertices = *hSimVertices;
  std::unordered_map<int, int> trackid_to_track_index;
  DecayChain decay;
  int idx = 0;

  IfLogDebug(DEBUG, messageCategory_) << " TRACKS" << std::endl;
  for (auto const &t : tracks) {
    IfLogDebug(DEBUG, messageCategory_) << " " << idx << "\t" << t.trackId() << "\t" << t << std::endl;
    trackid_to_track_index[t.trackId()] = idx;
    idx++;
  }

  /**
  Build the main decay graph and assign the SimTrack to each edge. The graph
  built here will only contain the particles that have a decay vertex
  associated to them. In order to recover also the particles that will not
  decay, we need to keep track of the SimTrack used here and add, a-posteriori,
  the ones not used, associating a ghost vertex (starting from the highest
  simulated vertex number), in order to build the edge and identify them
  immediately as stable (i.e. not decayed).

  To take into account the multi-bremsstrahlung effects in which a single
  particle is emitting photons in different vertices **keeping the same
  track index**, we also collapsed those vertices into 1 unique vertex. The
  other approach of fully representing the decay chain keeping the same
  track index would have the problem of over-counting the contributions of
  that track, especially in terms of hits.

  The 2 auxiliary vectors are structured as follow:

  1. used_sim_tracks is a vector that has the same size as the overall
     number of simulated tracks. The associated integer is the vertexId of
     the **decaying vertex for that track**.
  2. collapsed_vertices is a vector that has the same size as the overall
     number of simulated vertices. The vector's index is the vertexId
     itself, the associated value is the vertexId of the vertex on which
     this should collapse.
  */
  idx = 0;
  std::vector<int> used_sim_tracks(tracks.size(), 0);
  std::vector<int> collapsed_vertices(vertices.size(), 0);
  IfLogDebug(DEBUG, messageCategory_) << " VERTICES" << std::endl;
  for (auto const &v : vertices) {
    IfLogDebug(DEBUG, messageCategory_) << " " << idx++ << "\t" << v << std::endl;
    if (v.parentIndex() != -1) {
      auto trk_idx = trackid_to_track_index[v.parentIndex()];
      auto origin_vtx = tracks[trk_idx].vertIndex();
      if (used_sim_tracks[trk_idx]) {
        // collapse the vertex into the original first vertex we saw associated
        // to this track. Omit adding the edge in order to avoid double
        // counting of the very same particles  and its associated hits.
        collapsed_vertices[v.vertexId()] = used_sim_tracks[trk_idx];
        continue;
      }
      // Perform the actual vertex collapsing, if needed.
      if (collapsed_vertices[origin_vtx])
        origin_vtx = collapsed_vertices[origin_vtx];
      add_edge(origin_vtx,
               v.vertexId(),
               EdgeProperty(&tracks[trk_idx], simTrackDetIdEnergyMap[v.parentIndex()].size(), 0),
               decay);
      used_sim_tracks[trk_idx] = v.vertexId();
    }
  }
  // Build the motherParticle property to each vertex
  auto const &vertexMothersProp = get(vertex_name, decay);
  // Now recover the particles that did not decay. Append them with an index
  // bigger than the size of the generated vertices.
  int offset = vertices.size();
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!used_sim_tracks[i]) {
      auto origin_vtx = tracks[i].vertIndex();
      // Perform the actual vertex collapsing, if needed.
      if (collapsed_vertices[origin_vtx])
        origin_vtx = collapsed_vertices[origin_vtx];
      add_edge(
          origin_vtx, offset, EdgeProperty(&tracks[i], simTrackDetIdEnergyMap[tracks[i].trackId()].size(), 0), decay);
      // The properties for "fake" vertices associated to stable particles have
      // to be set inside this loop, since they do not belong to the vertices
      // collection and would be skipped by that loop (coming next)
      put(vertexMothersProp, offset, VertexProperty(&tracks[i], 0));
      offset++;
    }
  }
  for (auto const &v : vertices) {
    if (v.parentIndex() != -1) {
      // Skip collapsed_vertices
      if (collapsed_vertices[v.vertexId()])
        continue;
      put(vertexMothersProp, v.vertexId(), VertexProperty(&tracks[trackid_to_track_index[v.parentIndex()]], 0));
    }
  }
  SimHitsAccumulator_dfs_visitor vis;
  depth_first_search(decay, visitor(vis));
  CaloParticle_dfs_visitor caloParticleCreator(
      output_,
      m_caloParticles,
      m_simHitBarcodeToIndex,
      simTrackDetIdEnergyMap,
      [&](EdgeProperty &edge_property) -> bool {
        // Apply selection on SimTracks in order to promote them to be
        // CaloParticles. The function returns TRUE if the particle satisfies
        // the selection, FALSE otherwise. Therefore the correct logic to select
        // the particle is to ask for TRUE as return value.
        return (edge_property.cumulative_simHits != 0 and !edge_property.simTrack->noGenpart() and
                edge_property.simTrack->momentum().E() > minEnergy_ and
                std::abs(edge_property.simTrack->momentum().Eta()) < maxPseudoRapidity_);
      });
  depth_first_search(decay, visitor(caloParticleCreator));

#if DEBUG
  boost::write_graphviz(std::cout,
                        decay,
                        make_label_writer(make_transform_value_property_map(&graphviz_vertex, get(vertex_name, decay))),
                        make_label_writer(make_transform_value_property_map(&graphviz_edge, get(edge_weight, decay))));
#endif
}

template <class T>
void CaloTruthAccumulator::fillSimHits(std::vector<std::pair<DetId, const PCaloHit *>> &returnValue,
                                       std::unordered_map<int, std::map<int, float>> &simTrackDetIdEnergyMap,
                                       const T &event,
                                       const edm::EventSetup &setup) {
  for (auto const &collectionTag : collectionTags_) {
    edm::Handle<std::vector<PCaloHit>> hSimHits;
    const bool isHcal = (collectionTag.instance().find("HcalHits") != std::string::npos);
    event.getByLabel(collectionTag, hSimHits);

    for (auto const &simHit : *hSimHits) {
      DetId id(0);

      //Relabel as necessary for HGCAL
      if (doHGCAL) {
        const uint32_t simId = simHit.id();
        if (geometryType_ == 1) {
          // no test numbering in new geometry
          id = simId;
        } else if (isHcal) {
          HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd_);
          if (hid.subdet() == HcalEndcap)
            id = hid;
        } else {
          int subdet, layer, cell, sec, subsec, zp;
          HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
          const HGCalDDDConstants *ddd = hgddd_[subdet - 3];
          std::pair<int, int> recoLayerCell = ddd->simToReco(cell, layer, sec, hgtopo_[subdet - 3]->detectorType());
          cell = recoLayerCell.first;
          layer = recoLayerCell.second;
          // skip simhits with bad barcodes or non-existant layers
          if (layer == -1 || simHit.geantTrackId() == 0)
            continue;
          id = HGCalDetId((ForwardSubdetector)subdet, zp, layer, subsec, sec, cell);
        }
      } else {
        id = simHit.id();
        //Relabel all HCAL hits
        if (isHcal) {
          HcalDetId hid = HcalHitRelabeller::relabel(simHit.id(), hcddd_);
          id = hid;
        }
      }

      if (id == DetId(0)) {
        continue;
      }
      if (simHit.geantTrackId() == 0) {
        continue;
      }

      returnValue.emplace_back(id, &simHit);
      simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();
      m_detIdToTotalSimEnergy[id.rawId()] += simHit.energy();
    }
  }  // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR(CaloTruthAccumulator);
