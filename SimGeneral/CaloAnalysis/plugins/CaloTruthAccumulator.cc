#define DEBUG false 
#if DEBUG
// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// BOOST GRAPH LIBRARY
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>

#if DEBUG
#pragma GCC diagnostic pop
#endif

#include <iterator>
#include <numeric>  // for std::accumulate
#include <unordered_map>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducerBase.h"
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
}

using boost::adjacency_list;
using boost::directedS;
using boost::listS;
using boost::vecS;
using boost::property;
using boost::edge;
using boost::edge_weight_t;
using boost::edge_weight;
using boost::add_edge;
using boost::vertex;
using boost::vertex_name_t;
using boost::vertex_name;

/* GRAPH DEFINITIONS

   The graphs represent the full decay chain.

   The parent-child relationship is the natural one, following "time".

   Each edge has a property (edge_weight_t) that holds a const pointer to the
   SimTrack that connects the 2 vertices of the edge, the number of simHits
   associated to that simTrack and the cumulative number of simHits of itself
   and of all its children. Only simHits within the selected detectors are
   taken into account. The cumulative property is filled during the dfs
   exploration of the graph: if not explored the number is 0.

   Each vertex has a property (vertex_name_t) that holds a const pointer to the
   SimTrack that originated that vertex and the cumulative number of simHits of
   all its outgoing edges. The cumulative property is filled during the dfs
   exploration of the graph: if not explored the number is 0.

   Stable particles are recovered/added in a second iterations and are linked
   to ghost vertices with an offset starting from the highest generated vertex.

   Multiple decays of a single particle that retains its original trackId are
   merged into one unique vertex (the first encountered) in order to avoid
   multiple counting of its associated simHits (if any).

*/
struct EdgeProperty {
  EdgeProperty(const SimTrack* t, int h, int c) : simTrack(t), simHits(h), cumulative_simHits(c) {}
  const SimTrack* simTrack;
  int simHits;
  int cumulative_simHits;
};

struct VertexProperty {
  VertexProperty() : simTrack(nullptr), cumulative_simHits(0) {}
  VertexProperty(const SimTrack* t, int c) : simTrack(t), cumulative_simHits(c) {}
  VertexProperty(const VertexProperty& other)
      : simTrack(other.simTrack), cumulative_simHits(other.cumulative_simHits) {}
  const SimTrack* simTrack;
  int cumulative_simHits;
};

using EdgeParticleClustersProperty = property<edge_weight_t, EdgeProperty>;
using VertexMotherParticleProperty = property<vertex_name_t, VertexProperty>;
using DecayChain = adjacency_list<listS, vecS, directedS, VertexMotherParticleProperty,
                                  EdgeParticleClustersProperty>;

class CaloTruthAccumulator : public DigiAccumulatorMixMod {
 public:
  explicit CaloTruthAccumulator(const edm::ParameterSet& config,
      edm::ProducerBase& mixMod,
      edm::ConsumesCollector& iC);

 private:
  void initializeEvent(const edm::Event& event, const edm::EventSetup& setup) override;
  void accumulate(const edm::Event& event, const edm::EventSetup& setup) override;
  void accumulate(const PileUpEventPrincipal& event, const edm::EventSetup& setup,
                  edm::StreamID const&) override;
  void finalizeEvent(edm::Event& event, const edm::EventSetup& setup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi,
                            edm::EventSetup const& setup) override;

  /** @brief Both forms of accumulate() delegate to this templated method. */
  template <class T>
  void accumulateEvent(const T& event, const edm::EventSetup& setup,
                       const edm::Handle<edm::HepMCProduct>& hepMCproduct);

  /** @brief Fills the supplied vector with pointers to the SimHits, checking for bad modules if
   * required */
  template <class T>
  void fillSimHits(std::vector<std::pair<DetId, const PCaloHit*> >& returnValue,
                   std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap,
                   const T& event, const edm::EventSetup& setup);

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
  edm::Handle<std::vector<SimTrack> > hSimTracks;
  edm::Handle<std::vector<SimVertex> > hSimVertices;
  edm::Handle<std::vector<reco::GenParticle> > hGenParticles;

  std::vector<edm::InputTag> collectionTags_;
  edm::InputTag genParticleLabel_;
  /// Needed to add HepMC::GenVertex to SimVertex
  edm::InputTag hepMCproductLabel_;

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

    void swap(calo_particles& oth) {
      sc_start_.swap(oth.sc_start_);
      sc_stop_.swap(oth.sc_stop_);
    }

    void clear() {
      sc_start_.clear();
      sc_stop_.clear();
    }
  };

 private:
  const HGCalTopology* hgtopo_[3] = {nullptr,nullptr,nullptr};
  const HGCalDDDConstants* hgddd_[3] = {nullptr,nullptr,nullptr};
  const HcalDDDRecConstants* hcddd_ = nullptr;
  OutputCollections output_;
  calo_particles m_caloParticles;
  //geometry type (0 pre-TDR; 1 TDR)
  int geometryType_;
};

/* Graph utility functions */

namespace {
template <typename Edge, typename Graph, typename Visitor>
void accumulateSimHits_edge(Edge& e, const Graph& g, Visitor* v) {
  auto const edge_property = get(edge_weight, g, e);
  v->total_simHits += edge_property.simHits;
  IfLogDebug(DEBUG, messageCategoryGraph_)
      << " Examining edges " << e << " --> particle " << edge_property.simTrack->type() << "("
      << edge_property.simTrack->trackId() << ")"
      << " with SimClusters: " << edge_property.simHits
      << " Accumulated SimClusters: " << v->total_simHits << std::endl;
}
template <typename Vertex, typename Graph>
void print_vertex(Vertex& u, const Graph& g) {
  auto const vertex_property = get(vertex_name, g, u);
  IfLogDebug(DEBUG, messageCategoryGraph_) << " At " << u;
  // The Mother of all vertices has **no** SimTrack associated.
  if (vertex_property.simTrack)
    IfLogDebug(DEBUG, messageCategoryGraph_) << " [" << vertex_property.simTrack->type() << "]"
                                        << "(" << vertex_property.simTrack->trackId() << ")";
  IfLogDebug(DEBUG, messageCategoryGraph_) << std::endl;
}

// Graphviz output functions will only be generated in DEBUG mode
#if DEBUG
std::string graphviz_vertex(const VertexProperty& v) {
  std::ostringstream oss;
  oss << "{id: " << (v.simTrack ? v.simTrack->trackId() : 0)
      << ",\\ntype: " << (v.simTrack ? v.simTrack->type() : 0) << ",\\nchits: " << v.cumulative_simHits
      << "}";
  return oss.str();
}

std::string graphviz_edge(const EdgeProperty& e) {
  std::ostringstream oss;
  oss << "[" << (e.simTrack ? e.simTrack->trackId() : 0) << ","
      << (e.simTrack ? e.simTrack->type() : 0)
      << "," << e.simHits
      << "," << e.cumulative_simHits << "]";
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
    IfLogDebug(DEBUG, messageCategoryGraph_)
        << " Finished edge: " << e << " Track id: " << get(edge_weight, g, e).simTrack->trackId()
        << " has accumulated " << cumulative << " hits" << std::endl;
    IfLogDebug(DEBUG, messageCategoryGraph_)
        << " SrcVtx: " << src << "\t" << get(vertex_name, g, src).simTrack << "\t"
        << get(vertex_name, g, src).cumulative_simHits << std::endl;
    IfLogDebug(DEBUG, messageCategoryGraph_)
        << " TrgVtx: " << trg << "\t" << get(vertex_name, g, trg).simTrack << "\t"
        << get(vertex_name, g, trg).cumulative_simHits << std::endl;
  }
};

using Selector = std::function<bool(EdgeProperty&)>;

class CaloParticle_dfs_visitor : public boost::default_dfs_visitor {
 public:
  CaloParticle_dfs_visitor(CaloTruthAccumulator::OutputCollections& output,
                           CaloTruthAccumulator::calo_particles& caloParticles,
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
    //    if (u == 0) return;
    print_vertex(u, g);
    auto const vertex_property = get(vertex_name, g, u);
    if (!vertex_property.simTrack) return;
    auto trackIdx = vertex_property.simTrack->trackId();
    IfLogDebug(DEBUG, messageCategoryGraph_)
        << " Found " << simHitBarcodeToIndex_.count(trackIdx) << " associated simHits" << std::endl;
    if (simHitBarcodeToIndex_.count(trackIdx)) {
      output_.pSimClusters->emplace_back(*vertex_property.simTrack);
      auto& simcluster = output_.pSimClusters->back();
      std::unordered_map<uint32_t, float> acc_energy;
      for (auto const& hit_and_energy : simTrackDetIdEnergyMap_[trackIdx]) {
        acc_energy[hit_and_energy.first] += hit_and_energy.second;
      }
      for (auto const& hit_and_energy : acc_energy) {
        simcluster.addRecHitAndFraction(hit_and_energy.first, hit_and_energy.second);
      }
    }
  }
  template <typename Edge, typename Graph>
  void examine_edge(Edge e, const Graph& g) {
    auto src = source(e, g);
    auto vertex_property = get(vertex_name, g, src);
    if (src == 0 or (vertex_property.simTrack == nullptr)) {
      auto edge_property = get(edge_weight, g, e);
      IfLogDebug(DEBUG, messageCategoryGraph_)
        << "Considering CaloParticle: " << edge_property.simTrack->trackId();
      if (selector_(edge_property)) {
        IfLogDebug(DEBUG, messageCategoryGraph_)
          << "Adding CaloParticle: " << edge_property.simTrack->trackId();
        output_.pCaloParticles->emplace_back(*(edge_property.simTrack));
        caloParticles_.sc_start_.push_back(output_.pSimClusters->size());
      }
    }
  }

  template <typename Edge, typename Graph>
  void finish_edge(Edge e, const Graph& g) {
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
  CaloTruthAccumulator::OutputCollections& output_;
  CaloTruthAccumulator::calo_particles& caloParticles_;
  std::unordered_multimap<Barcode_t, Index_t>& simHitBarcodeToIndex_;
  std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap_;
  Selector selector_;
};
}

CaloTruthAccumulator::CaloTruthAccumulator(const edm::ParameterSet& config,
                                                             edm::ProducerBase& mixMod,
                                                             edm::ConsumesCollector& iC)
    : messageCategory_("CaloTruthAccumulator"),
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
      maxPseudoRapidity_(config.getParameter<double>("MaxPseudoRapidity")),
      premixStage1_(config.getParameter<bool>("premixStage1")),
      geometryType_(-1)
{
  mixMod.produces<SimClusterCollection>("MergedCaloTruth");
  mixMod.produces<CaloParticleCollection>("MergedCaloTruth");
  if(premixStage1_) {
    mixMod.produces<std::vector<std::pair<unsigned int, float> > >("MergedCaloTruth");
  }

  iC.consumes<std::vector<SimTrack> >(simTrackLabel_);
  iC.consumes<std::vector<SimVertex> >(simVertexLabel_);
  iC.consumes<std::vector<reco::GenParticle> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(genParticleLabel_);
  iC.consumes<std::vector<int> >(hepMCproductLabel_);

  // Fill the collection tags
  const edm::ParameterSet& simHitCollectionConfig = config.getParameterSet("simHitCollections");
  std::vector<std::string> parameterNames = simHitCollectionConfig.getParameterNames();

  for (auto const& parameterName : parameterNames) {
    std::vector<edm::InputTag> tags =
        simHitCollectionConfig.getParameter<std::vector<edm::InputTag> >(parameterName);
    collectionTags_.insert(collectionTags_.end(), tags.begin(), tags.end());
  }

  for (auto const& collectionTag : collectionTags_) {
    iC.consumes<std::vector<PCaloHit> >(collectionTag);
  }
}

void CaloTruthAccumulator::beginLuminosityBlock(edm::LuminosityBlock const& iLumiBlock,
                                                         const edm::EventSetup& iSetup) {
  return;
  edm::ESHandle<CaloGeometry> geom;
  iSetup.get<CaloGeometryRecord>().get(geom);
  const HGCalGeometry *eegeom = nullptr, *fhgeom = nullptr, *bhgeomnew = nullptr;
  const HcalGeometry* bhgeom = nullptr;

  eegeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::HGCalEE,ForwardSubdetector::ForwardEmpty));
  //check if it's the new geometry
  if(eegeom){
    geometryType_ = 1;
    fhgeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::HGCalHSi,ForwardSubdetector::ForwardEmpty));
    bhgeomnew = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::HGCalHSc,ForwardSubdetector::ForwardEmpty));
  }
  else {
    geometryType_ = 0;
    eegeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCEE));
    fhgeom = static_cast<const HGCalGeometry*>(geom->getSubdetectorGeometry(DetId::Forward, HGCHEF));
    bhgeom = static_cast<const HcalGeometry*>(geom->getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
  }
  hgtopo_[0] = &(eegeom->topology());
  hgtopo_[1] = &(fhgeom->topology());
  if(bhgeomnew) hgtopo_[2] = &(bhgeomnew->topology());

  for (unsigned i = 0; i < 3; ++i) {
    if(hgtopo_[i]) hgddd_[i] = &(hgtopo_[i]->dddConstants());
  }

  if(bhgeom) hcddd_ = bhgeom->topology().dddConstants();
}

void CaloTruthAccumulator::initializeEvent(edm::Event const& event,
                                                    edm::EventSetup const& setup) {
  output_.pSimClusters.reset(new SimClusterCollection());
  output_.pCaloParticles.reset(new CaloParticleCollection());

  m_detIdToTotalSimEnergy.clear();
}

/** Create handle to edm::HepMCProduct here because event.getByLabel with
    edm::HepMCProduct only works for edm::Event but not for
    PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call
    T::value_type and T::iterator (where T is the type of the object one wants
    to get a handle to) which is only implemented for container-like objects
    like std::vector but not for edm::HepMCProduct!
*/
void CaloTruthAccumulator::accumulate(edm::Event const& event,
                                               edm::EventSetup const& setup) {
  edm::Handle<edm::HepMCProduct> hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);

  edm::LogInfo(messageCategory_) << " CaloTruthAccumulator::accumulate (signal)";
  accumulateEvent(event, setup, hepmc);
}

void CaloTruthAccumulator::accumulate(PileUpEventPrincipal const& event,
                                               edm::EventSetup const& setup, edm::StreamID const&) {
  if (event.bunchCrossing() >= -static_cast<int>(maximumPreviousBunchCrossing_) &&
      event.bunchCrossing() <= static_cast<int>(maximumSubsequentBunchCrossing_)) {
    // simply create empty handle as we do not have a HepMCProduct in PU anyway
    edm::Handle<edm::HepMCProduct> hepmc;
    edm::LogInfo(messageCategory_)
        << " CaloTruthAccumulator::accumulate (pileup) bunchCrossing="
        << event.bunchCrossing();
    accumulateEvent(event, setup, hepmc);
  } else {
    edm::LogInfo(messageCategory_)
        << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
  }
}

void CaloTruthAccumulator::finalizeEvent(edm::Event& event, edm::EventSetup const& setup) {
  edm::LogInfo(messageCategory_) << "Adding " << output_.pSimClusters->size()
                                 << " SimParticles and " << output_.pCaloParticles->size()
                                 << " CaloParticles to the event.";

  // We need to normalize the hits and energies into hits and fractions (since
  // we have looped over all pileup events)
  // For premixing stage1 we keep the energies, they will be normalized to fractions in stage2

  if(premixStage1_) {
    auto totalEnergies = std::make_unique<std::vector<std::pair<unsigned int, float> > >();
    totalEnergies->reserve(m_detIdToTotalSimEnergy.size());
    std::copy(m_detIdToTotalSimEnergy.begin(), m_detIdToTotalSimEnergy.end(), std::back_inserter(*totalEnergies));
    std::sort(totalEnergies->begin(), totalEnergies->end());
    event.put(std::move(totalEnergies), "MergedCaloTruth");
  }
  else {
    for (auto& sc : *(output_.pSimClusters)) {
      auto hitsAndEnergies = sc.hits_and_fractions();
      for (auto& hAndE : hitsAndEnergies)
        sc.addRecHitAndEnergy(hAndE.first, hAndE.second); 
      sc.clearHitsAndFractions();
      for (auto& hAndE : hitsAndEnergies) {
        const float totalenergy = m_detIdToTotalSimEnergy[hAndE.first];
        float fraction = 0.;
        if (totalenergy > 0)
          fraction = hAndE.second / totalenergy;
        else
          edm::LogWarning(messageCategory_) << "TotalSimEnergy for hit " << hAndE.first
                                            << " is 0! The fraction for this hit cannot be computed.";
        sc.addRecHitAndFraction(hAndE.first, fraction);
      }
    }
  }

  // save the SimCluster orphan handle so we can fill the calo particles
  auto scHandle = event.put(std::move(output_.pSimClusters), "MergedCaloTruth");

  // now fill the calo particles
  for (unsigned i = 0; i < output_.pCaloParticles->size(); ++i) {
    auto& cp = (*output_.pCaloParticles)[i];
    auto& g4Tracks_tmp = cp.g4Tracks();
    edm::Ref<std::vector<reco::GenParticle> > ref_gen(hGenParticles, g4Tracks_tmp[0].genpartIndex()-1);
    cp.addGenParticle(ref_gen);
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
void CaloTruthAccumulator::accumulateEvent(
    const T& event, const edm::EventSetup& setup,
    const edm::Handle<edm::HepMCProduct>& hepMCproduct) {

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

  auto const& tracks = *hSimTracks;
  auto const& vertices = *hSimVertices;
  std::unordered_map<int, int> trackid_to_track_index;
  DecayChain decay;
  int idx = 0;

  std::cout  << " TRACKS" << std::endl;
  for (auto const& t : tracks) {
    //std::cout 
    //    << " " << idx << "\t" << t.trackId() << "\t" << t << std::endl;
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
  std::cout  << " VERTICES" << std::endl;
  for (auto const& v : vertices) {
    //std::cout  << " " << idx++ << "\t" << v << std::endl;
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
      if (collapsed_vertices[origin_vtx]) origin_vtx = collapsed_vertices[origin_vtx];
      add_edge(origin_vtx, v.vertexId(),
               EdgeProperty(&tracks[trk_idx], simTrackDetIdEnergyMap[v.parentIndex()].size(), 0),
               decay);
      used_sim_tracks[trk_idx] = v.vertexId();
    }
  }
  // Build the motherParticle property to each vertex
  auto const& vertexMothersProp = get(vertex_name, decay);
  // Now recover the particles that did not decay. Append them with an index
  // bigger than the size of the generated vertices.
  int offset = vertices.size();
  for (size_t i = 0; i < tracks.size(); ++i) {
    if (!used_sim_tracks[i]) {
      auto origin_vtx = tracks[i].vertIndex();
      // Perform the actual vertex collapsing, if needed.
      if (collapsed_vertices[origin_vtx]) origin_vtx = collapsed_vertices[origin_vtx];
      add_edge(origin_vtx, offset,
               EdgeProperty(&tracks[i], simTrackDetIdEnergyMap[tracks[i].trackId()].size(), 0),
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
      // Skip collapsed_vertices
      if (collapsed_vertices[v.vertexId()]) continue;
      put(vertexMothersProp, v.vertexId(),
          VertexProperty(&tracks[trackid_to_track_index[v.parentIndex()]], 0));
    }
  }
  SimHitsAccumulator_dfs_visitor vis;
  depth_first_search(decay, visitor(vis));
  CaloParticle_dfs_visitor caloParticleCreator(
      output_, m_caloParticles, m_simHitBarcodeToIndex, simTrackDetIdEnergyMap,
      [&](EdgeProperty& edge_property) -> bool {
        // Apply selection on SimTracks in order to promote them to be CaloParticles.
        // The function returns TRUE if the particle satisfies the selection, FALSE otherwise.
        // Therefore the correct logic to select the particle is to ask for TRUE as return value.
        return (edge_property.cumulative_simHits != 0 and !edge_property.simTrack->noGenpart() and
                edge_property.simTrack->momentum().E() > minEnergy_ and
                std::abs(edge_property.simTrack->momentum().Eta()) < maxPseudoRapidity_);
      });
  depth_first_search(decay, visitor(caloParticleCreator));

#if DEBUG
  boost::write_graphviz(std::cout, decay, make_label_writer(make_transform_value_property_map(
                                              &graphviz_vertex, get(vertex_name, decay))),
                        make_label_writer(make_transform_value_property_map(
                            &graphviz_edge, get(edge_weight, decay))));
#endif
}

template <class T>
void CaloTruthAccumulator::fillSimHits(
    std::vector<std::pair<DetId, const PCaloHit*> >& returnValue,
    std::unordered_map<int, std::map<int, float> >& simTrackDetIdEnergyMap, const T& event,
    const edm::EventSetup& setup) {
  for (auto const& collectionTag : collectionTags_) {
    edm::Handle<std::vector<PCaloHit> > hSimHits;
    const bool isHcal = (collectionTag.instance().find("HcalHits") != std::string::npos);
    event.getByLabel(collectionTag, hSimHits);
    for (auto const& simHit : *hSimHits) {
      DetId id = simHit.id();
      const uint32_t simId = simHit.id();
//      if (geometryType_==1) {
//        //no test numbering in new geometry
//        id = simId;
//      }
//      else if (isHcal) {
//        HcalDetId hid = HcalHitRelabeller::relabel(simId, hcddd_);
//        if (hid.subdet() == HcalEndcap) id = hid;
//      } else {
//        int subdet, layer, cell, sec, subsec, zp;
//        HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);
//        const HGCalDDDConstants* ddd = hgddd_[subdet - 3];
//        std::pair<int, int> recoLayerCell =
//            ddd->simToReco(cell, layer, sec, hgtopo_[subdet - 3]->detectorType());
//        cell = recoLayerCell.first;
//        layer = recoLayerCell.second;
//        // skip simhits with bad barcodes or non-existant layers
//        if (layer == -1 || simHit.geantTrackId() == 0) continue;
//        id = HGCalDetId((ForwardSubdetector)subdet, zp, layer, subsec, sec, cell);
//      }

      if (DetId(0) == id) continue;

      uint32_t detId = id.rawId();
      returnValue.emplace_back(id, &simHit);
      simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();

      m_detIdToTotalSimEnergy[detId] += simHit.energy();
    }
  }  // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR(CaloTruthAccumulator);
