// Author: Aurora Perego, Fabio Cossutti - aurora.perego@cern.ch, fabio.cossutti@ts.infn.it
// Date: 05/2023

#define DEBUG false

#if DEBUG
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <iterator>
#include <exception>
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

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "SimDataFormats/CaloAnalysis/interface/MtdCaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdCaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimTrackster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimTracksterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "SimGeneral/MixingModule/interface/DecayGraph.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

namespace {
  using Index_t = unsigned;
  using Barcode_t = int;
  const std::string messageCategoryGraph_("MtdTruthAccumulatorGraphProducer");
}  // namespace

class MtdTruthAccumulator : public DigiAccumulatorMixMod {
public:
  explicit MtdTruthAccumulator(const edm::ParameterSet &config, edm::ProducesCollector, edm::ConsumesCollector &iC);

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
  void fillSimHits(std::vector<std::pair<uint64_t, const PSimHit *>> &returnValue,
                   std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdEnergyMap,
                   std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdTimeMap,
                   const T &event,
                   const edm::EventSetup &setup);

  const std::string messageCategory_;

  std::unordered_map<uint64_t, float> m_detIdToTotalSimEnergy;  // keep track of cell normalizations
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

  const unsigned int bunchSpacing_;

  const edm::InputTag simTrackLabel_;
  const edm::InputTag simVertexLabel_;
  edm::Handle<std::vector<SimTrack>> hSimTracks;
  edm::Handle<std::vector<SimVertex>> hSimVertices;

  std::vector<edm::InputTag> collectionTags_;
  edm::InputTag genParticleLabel_;
  /// Needed to add HepMC::GenVertex to SimVertex
  edm::InputTag hepMCproductLabel_;
  const edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  // edm::ESWatcher<MTDDigiGeometryRecord> geomWatcher_;

  mtd::MTDGeomUtil geomTools_;

  const double minEnergy_, maxPseudoRapidity_;
  const bool premixStage1_;

  bool isEtl_;

public:
  struct OutputCollections {
    std::unique_ptr<MtdSimClusterCollection> pSimClusters;
    std::unique_ptr<MtdCaloParticleCollection> pCaloParticles;
    std::unique_ptr<MtdSimLayerClusterCollection> pMtdSimLayerClusters;
    std::unique_ptr<MtdSimTracksterCollection> pMtdSimTracksters;
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
  const MTDGeometry *geom = nullptr;
  const MTDTopology *topology = nullptr;
  OutputCollections output_;
  calo_particles m_caloParticles;
};

/* Graph utility functions */

namespace {
  class CaloParticle_dfs_visitor : public boost::default_dfs_visitor {
  public:
    CaloParticle_dfs_visitor(MtdTruthAccumulator::OutputCollections &output,
                             MtdTruthAccumulator::calo_particles &caloParticles,
                             std::unordered_multimap<Barcode_t, Index_t> &simHitBarcodeToIndex,
                             std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdEnergyMap,
                             std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdTimeMap,
                             std::unordered_map<uint32_t, float> &vertex_time_map,
                             Selector selector)
        : output_(output),
          caloParticles_(caloParticles),
          simHitBarcodeToIndex_(simHitBarcodeToIndex),
          simTrackDetIdEnergyMap_(simTrackDetIdEnergyMap),
          simTrackDetIdTimeMap_(simTrackDetIdTimeMap),
          vertex_time_map_(vertex_time_map),
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
        std::unordered_map<uint64_t, float> acc_energy;
        for (auto const &hit_and_energy : simTrackDetIdEnergyMap_[trackIdx]) {
          acc_energy[hit_and_energy.first] += hit_and_energy.second;
        }
        for (auto const &hit_and_energy : acc_energy) {
          simcluster.addHitAndFraction(hit_and_energy.first, hit_and_energy.second);
          simcluster.addHitEnergy(hit_and_energy.second);
          simcluster.addHitTime(simTrackDetIdTimeMap_[simcluster.g4Tracks()[0].trackId()][hit_and_energy.first]);
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
          output_.pCaloParticles->back().addSimTime(vertex_time_map_[(edge_property.simTrack)->vertIndex()]);
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
    MtdTruthAccumulator::OutputCollections &output_;
    MtdTruthAccumulator::calo_particles &caloParticles_;
    std::unordered_multimap<Barcode_t, Index_t> &simHitBarcodeToIndex_;
    std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdEnergyMap_;
    std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdTimeMap_;
    std::unordered_map<uint32_t, float> &vertex_time_map_;
    Selector selector_;
  };
}  // namespace

MtdTruthAccumulator::MtdTruthAccumulator(const edm::ParameterSet &config,
                                         edm::ProducesCollector producesCollector,
                                         edm::ConsumesCollector &iC)
    : messageCategory_("MtdTruthAccumulator"),
      maximumPreviousBunchCrossing_(config.getParameter<unsigned int>("maximumPreviousBunchCrossing")),
      maximumSubsequentBunchCrossing_(config.getParameter<unsigned int>("maximumSubsequentBunchCrossing")),
      bunchSpacing_(config.getParameter<unsigned int>("bunchspace")),
      simTrackLabel_(config.getParameter<edm::InputTag>("simTrackCollection")),
      simVertexLabel_(config.getParameter<edm::InputTag>("simVertexCollection")),
      collectionTags_(),
      genParticleLabel_(config.getParameter<edm::InputTag>("genParticleCollection")),
      hepMCproductLabel_(config.getParameter<edm::InputTag>("HepMCProductLabel")),
      geomToken_(iC.esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
      mtdtopoToken_(iC.esConsumes<MTDTopology, MTDTopologyRcd>()),
      minEnergy_(config.getParameter<double>("MinEnergy")),
      maxPseudoRapidity_(config.getParameter<double>("MaxPseudoRapidity")),
      premixStage1_(config.getParameter<bool>("premixStage1")) {
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
    iC.consumes<std::vector<PSimHit>>(collectionTag);
    isEtl_ = (collectionTag.instance().find("FastTimerHitsEndcap") != std::string::npos);
  }

  producesCollector.produces<MtdSimClusterCollection>("MergedMtdTruth");
  producesCollector.produces<MtdSimLayerClusterCollection>("MergedMtdTruthLC");
  producesCollector.produces<MtdSimTracksterCollection>("MergedMtdTruthST");
  producesCollector.produces<MtdCaloParticleCollection>("MergedMtdTruth");
  if (premixStage1_) {
    producesCollector.produces<std::vector<std::pair<uint64_t, float>>>("MergedMtdTruth");
  }
}

void MtdTruthAccumulator::initializeEvent(edm::Event const &event, edm::EventSetup const &setup) {
  output_.pSimClusters = std::make_unique<MtdSimClusterCollection>();
  output_.pCaloParticles = std::make_unique<MtdCaloParticleCollection>();

  output_.pMtdSimLayerClusters = std::make_unique<MtdSimLayerClusterCollection>();
  output_.pMtdSimTracksters = std::make_unique<MtdSimTracksterCollection>();

  m_detIdToTotalSimEnergy.clear();

  auto geometryHandle = setup.getTransientHandle(geomToken_);
  geom = geometryHandle.product();

  auto topologyHandle = setup.getTransientHandle(mtdtopoToken_);
  topology = topologyHandle.product();

  geomTools_.setGeometry(geom);
  geomTools_.setTopology(topology);
}

/** Create handle to edm::HepMCProduct here because event.getByLabel with
    edm::HepMCProduct only works for edm::Event but not for
    PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call
    T::value_type and T::iterator (where T is the type of the object one wants
    to get a handle to) which is only implemented for container-like objects
    like std::vector but not for edm::HepMCProduct!
*/
void MtdTruthAccumulator::accumulate(edm::Event const &event, edm::EventSetup const &setup) {
  edm::Handle<edm::HepMCProduct> hepmc;
  event.getByLabel(hepMCproductLabel_, hepmc);

  edm::LogInfo(messageCategory_) << " MtdTruthAccumulator::accumulate (signal)";
  accumulateEvent(event, setup, hepmc);
}

void MtdTruthAccumulator::accumulate(PileUpEventPrincipal const &event,
                                     edm::EventSetup const &setup,
                                     edm::StreamID const &) {
  if (event.bunchCrossing() >= -static_cast<int>(maximumPreviousBunchCrossing_) &&
      event.bunchCrossing() <= static_cast<int>(maximumSubsequentBunchCrossing_)) {
    // simply create empty handle as we do not have a HepMCProduct in PU anyway
    edm::Handle<edm::HepMCProduct> hepmc;
    edm::LogInfo(messageCategory_) << " MtdTruthAccumulator::accumulate (pileup) bunchCrossing="
                                   << event.bunchCrossing();
    accumulateEvent(event, setup, hepmc);
  } else {
    edm::LogInfo(messageCategory_) << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
  }
}

void MtdTruthAccumulator::finalizeEvent(edm::Event &event, edm::EventSetup const &setup) {
  using namespace geant_units::operators;

  edm::LogInfo(messageCategory_) << "Adding " << output_.pSimClusters->size() << " SimParticles and "
                                 << output_.pCaloParticles->size() << " CaloParticles to the event.";

  // We need to normalize the hits and energies into hits and fractions (since
  // we have looped over all pileup events)
  // For premixing stage1 we keep the energies, they will be normalized to
  // fractions in stage2

  if (premixStage1_) {
    auto totalEnergies = std::make_unique<std::vector<std::pair<uint64_t, float>>>();
    totalEnergies->reserve(m_detIdToTotalSimEnergy.size());
    std::copy(m_detIdToTotalSimEnergy.begin(), m_detIdToTotalSimEnergy.end(), std::back_inserter(*totalEnergies));
    std::sort(totalEnergies->begin(), totalEnergies->end());
    event.put(std::move(totalEnergies), "MergedMtdTruth");
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
        sc.addHitAndFraction(hAndE.first, fraction);
        sc.addHitEnergy(hAndE.second);
      }
    }
  }

#ifdef PRINT_DEBUG
  IfLogDebug(DEBUG, messageCategory_) << "SIMCLUSTERS LIST:" << std::endl;
  for (const auto &sc : *(output_.pSimClusters)) {
    IfLogDebug(DEBUG, messageCategory_) << std::fixed << std::setprecision(3) << "SimCluster from CP with:"
                                        << "\n  charge " << sc.charge() << "\n  pdgId  " << sc.pdgId() << "\n  energy "
                                        << sc.energy() << " GeV\n  eta    " << sc.eta() << "\n  phi    " << sc.phi()
                                        << "\n  number of cells = " << sc.hits_and_fractions().size() << std::endl;
    for (unsigned int i = 0; i < sc.hits_and_fractions().size(); ++i) {
      DetId id(sc.detIds_and_rows()[i].first);
      IfLogDebug(DEBUG, messageCategory_)
          << std::fixed << std::setprecision(3) << " hit " << id.rawId() << " on layer " << geomTools_.layer(id)
          << " module " << geomTools_.module(id) << " row " << (unsigned int)(sc.detIds_and_rows()[i].second).first
          << " col " << (unsigned int)(sc.detIds_and_rows()[i].second).second << " at time "
          << sc.hits_and_times()[i].second << " ns" << std::endl;
    }
    IfLogDebug(DEBUG, messageCategory_) << "--------------\n";
  }
  IfLogDebug(DEBUG, messageCategory_) << std::endl;
#endif

  // save the SimCluster orphan handle so we can fill the calo particles
  auto scHandle = event.put(std::move(output_.pSimClusters), "MergedMtdTruth");

  // reserve for the best case scenario: already splitted
  output_.pMtdSimLayerClusters->reserve(scHandle->size());
  output_.pMtdSimTracksters->reserve(scHandle->size());

  uint32_t SC_index = 0;
  uint32_t LC_index = 0;
  for (const auto &sc : *scHandle) {
    auto const &hAndF = sc.hits_and_fractions();
    auto const &hAndE = sc.hits_and_energies();
    auto const &hAndT = sc.hits_and_times();
    auto const &hAndR = sc.detIds_and_rows();
    // create a vector with the indices of the hits in the simCluster
    std::vector<int> indices(hAndF.size());
    std::iota(indices.begin(), indices.end(), 0);
    // sort the hits indices based on the unique indices created before
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return hAndF[a].first < hAndF[b].first; });

    // now split the sc: loop on the sorted indices and save the first hit in a
    // temporary simCluster. If the following hit is in the same module and row (column),
    // but next column (row) put it in the temporary simcluster as well, otherwise
    // put the temporary simcluster in the collection and start creating a new one
    std::vector<uint32_t> LC_indices;
    MtdSimLayerCluster tmpLC(sc.g4Tracks()[0]);
    int prev = indices[0];
    DetId prevId(hAndR[prev].first);

    float SimLCenergy = 0.;
    float SimLCx = 0., SimLCy = 0., SimLCz = 0.;

    auto push_back_hit = [&](const int &ind) {
      tmpLC.addHitAndFraction(hAndF[ind].first, hAndF[ind].second);
      tmpLC.addHitEnergy(hAndE[ind].second);
      tmpLC.addHitTime(hAndT[ind].second);
    };

    auto update_clu_info = [&](const int &ind) {
      double energy = hAndE[ind].second;
      auto position =
          geomTools_.position((DetId)hAndR[ind].first, (hAndR[ind].second).first, (hAndR[ind].second).second).first;
      SimLCenergy += energy;
      SimLCx += position.x() * energy;
      SimLCy += position.y() * energy;
      SimLCz += position.z() * energy;
    };

    auto push_back_clu = [&](const uint32_t &SC_index, uint32_t &LC_index) {
      tmpLC.addCluEnergy(SimLCenergy);
      LocalPoint SimLCpos(SimLCx / SimLCenergy, SimLCy / SimLCenergy, SimLCz / SimLCenergy);
      tmpLC.addCluLocalPos(SimLCpos);
      SimLCenergy = 0.;
      SimLCx = 0.;
      SimLCy = 0.;
      SimLCz = 0.;
      tmpLC.addCluIndex(SC_index);
      tmpLC.computeClusterTime();
      output_.pMtdSimLayerClusters->push_back(tmpLC);
      LC_indices.push_back(LC_index);
      LC_index++;
      tmpLC.clear();
    };

    // fill tmpLC with the first hit
    push_back_hit(prev);
    update_clu_info(prev);
    for (const auto &ind : indices) {
      if (ind == indices[0])
        continue;
      DetId id(hAndR[ind].first);
      if (geomTools_.isETL(id) != geomTools_.isETL(prevId) or geomTools_.layer(id) != geomTools_.layer(prevId) or
          geomTools_.module(id) != geomTools_.module(prevId) or
          ((hAndR[ind].second).first == (hAndR[prev].second).first and
           (hAndR[ind].second).second != (hAndR[prev].second).second + 1) or
          ((hAndR[ind].second).second == (hAndR[prev].second).second and
           (hAndR[ind].second).first != (hAndR[prev].second).first + 1)) {
        // the next hit is not adjacent to the previous one, put the current temporary cluster in the collection
        // and the hit will be put in an empty temporary cluster
        push_back_clu(SC_index, LC_index);
      }
      // add the hit to the temporary cluster
      push_back_hit(ind);
      update_clu_info(ind);
      prev = ind;
      DetId newId(hAndR[prev].first);
      prevId = newId;
    }
    // add the remaining temporary cluster to the collection
    push_back_clu(SC_index, LC_index);

    // now the simTrackster: find position and time of the first simHit
    // bc right now there is no method to ask the simTrack for pos/time
    // at MTD entrance
    float timeAtEntrance = 99.;
    uint32_t idAtEntrance = 0;
    for (uint32_t i = 0; i < (uint32_t)hAndT.size(); i++) {
      if (hAndT[i].second < timeAtEntrance) {
        timeAtEntrance = hAndT[i].second;
        idAtEntrance = i;
      }
    }

    // sort LCs in the SimTrackster by time
    auto &MtdSimLayerClusters = output_.pMtdSimLayerClusters;
    std::sort(LC_indices.begin(), LC_indices.end(), [&MtdSimLayerClusters](int i, int j) {
      return (*MtdSimLayerClusters)[i].simLCTime() < (*MtdSimLayerClusters)[j].simLCTime();
    });

    GlobalPoint posAtEntrance = geomTools_
                                    .position((DetId)hAndR[idAtEntrance].first,
                                              (hAndR[idAtEntrance].second).first,
                                              (hAndR[idAtEntrance].second).second)
                                    .second;
    output_.pMtdSimTracksters->emplace_back(sc, LC_indices, timeAtEntrance, posAtEntrance);
    SC_index++;
  }

#ifdef PRINT_DEBUG
  IfLogDebug(DEBUG, messageCategory_) << "SIMLAYERCLUSTERS LIST: \n";
  for (auto &sc : *output_.pMtdSimLayerClusters) {
    IfLogDebug(DEBUG, messageCategory_) << std::fixed << std::setprecision(3) << "SimLayerCluster with:"
                                        << "\n  CP charge " << sc.charge() << "\n  CP pdgId  " << sc.pdgId()
                                        << "\n  CP energy " << sc.energy() << " GeV\n  CP eta    " << sc.eta()
                                        << "\n  CP phi    " << sc.phi()
                                        << "\n  number of cells = " << sc.hits_and_fractions().size() << std::endl;
    for (unsigned int i = 0; i < sc.hits_and_fractions().size(); ++i) {
      DetId id(sc.detIds_and_rows()[i].first);
      IfLogDebug(DEBUG, messageCategory_)
          << std::fixed << std::setprecision(3) << " hit " << sc.detIds_and_rows()[i].first << " on layer "
          << geomTools_.layer(id) << " at time " << sc.hits_and_times()[i].second << " ns" << std::endl;
    }
    IfLogDebug(DEBUG, messageCategory_) << std::fixed << std::setprecision(3) << "  Cluster time " << sc.simLCTime()
                                        << " ns \n Cluster pos" << sc.simLCPos() << " cm\n"
                                        << std::fixed << std::setprecision(6) << " Cluster energy "
                                        << convertUnitsTo(0.001_MeV, sc.simLCEnergy()) << " MeV" << std::endl;
    IfLogDebug(DEBUG, messageCategory_) << "--------------\n";
  }
  IfLogDebug(DEBUG, messageCategory_) << std::endl;

  IfLogDebug(DEBUG, messageCategory_) << "SIMTRACKSTERS LIST: \n";
  for (auto &sc : *output_.pMtdSimTracksters) {
    IfLogDebug(DEBUG, messageCategory_) << std::fixed << std::setprecision(3) << "SimTrackster with:"
                                        << "\n  CP charge " << sc.charge() << "\n  CP pdgId  " << sc.pdgId()
                                        << "\n  CP energy " << sc.energy() << " GeV\n  CP eta    " << sc.eta()
                                        << "\n  CP phi    " << sc.phi()
                                        << "\n number of layer clusters = " << sc.numberOfClusters()
                                        << "\n time of first simhit " << sc.time() << " ns\n position of first simhit"
                                        << sc.position() << "cm" << std::endl;
    IfLogDebug(DEBUG, messageCategory_) << "  LCs indices: ";
    for (const auto &lc : sc.clusters())
      IfLogDebug(DEBUG, messageCategory_) << lc << ", ";
    IfLogDebug(DEBUG, messageCategory_) << "\n--------------\n";
  }
  IfLogDebug(DEBUG, messageCategory_) << std::endl;
#endif

  event.put(std::move(output_.pMtdSimLayerClusters), "MergedMtdTruthLC");
  event.put(std::move(output_.pMtdSimTracksters), "MergedMtdTruthST");

  // now fill the calo particles
  for (unsigned i = 0; i < output_.pCaloParticles->size(); ++i) {
    auto &cp = (*output_.pCaloParticles)[i];
    for (unsigned j = m_caloParticles.sc_start_[i]; j < m_caloParticles.sc_stop_[i]; ++j) {
      edm::Ref<MtdSimClusterCollection> ref(scHandle, j);
      cp.addSimCluster(ref);
    }
  }
  event.put(std::move(output_.pCaloParticles), "MergedMtdTruth");

  calo_particles().swap(m_caloParticles);

  std::unordered_map<uint64_t, float>().swap(m_detIdToTotalSimEnergy);
  std::unordered_multimap<Barcode_t, Index_t>().swap(m_simHitBarcodeToIndex);
}

template <class T>
void MtdTruthAccumulator::accumulateEvent(const T &event,
                                          const edm::EventSetup &setup,
                                          const edm::Handle<edm::HepMCProduct> &hepMCproduct) {
  edm::Handle<std::vector<reco::GenParticle>> hGenParticles;
  edm::Handle<std::vector<int>> hGenParticleIndices;

  event.getByLabel(simTrackLabel_, hSimTracks);
  event.getByLabel(simVertexLabel_, hSimVertices);

  event.getByLabel(genParticleLabel_, hGenParticles);
  event.getByLabel(genParticleLabel_, hGenParticleIndices);

  std::vector<std::pair<uint64_t, const PSimHit *>> simHitPointers;
  std::unordered_map<int, std::map<uint64_t, float>> simTrackDetIdEnergyMap;
  std::unordered_map<int, std::map<uint64_t, float>> simTrackDetIdTimeMap;
  fillSimHits(simHitPointers, simTrackDetIdEnergyMap, simTrackDetIdTimeMap, event, setup);

  // Clear maps from previous event fill them for this one
  m_simHitBarcodeToIndex.clear();
  for (unsigned int i = 0; i < simHitPointers.size(); ++i) {
    m_simHitBarcodeToIndex.emplace(simHitPointers[i].second->trackId(), i);
  }

  auto const &tracks = *hSimTracks;
  auto const &vertices = *hSimVertices;
  std::unordered_map<int, int> trackid_to_track_index;
  std::unordered_map<uint32_t, float> vertex_time_map;
  DecayChain decay;

  for (uint32_t i = 0; i < vertices.size(); i++) {
    vertex_time_map[i] = vertices[i].position().t() * 1e9 + event.bunchCrossing() * bunchSpacing_;
  }

  IfLogDebug(DEBUG, messageCategory_) << " TRACKS" << std::endl;
  int idx = 0;
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
      auto const trk_idx = trackid_to_track_index[v.parentIndex()];
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
      simTrackDetIdTimeMap,
      vertex_time_map,
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
void MtdTruthAccumulator::fillSimHits(std::vector<std::pair<uint64_t, const PSimHit *>> &returnValue,
                                      std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdEnergyMap,
                                      std::unordered_map<int, std::map<uint64_t, float>> &simTrackDetIdTimeMap,
                                      const T &event,
                                      const edm::EventSetup &setup) {
  using namespace geant_units::operators;
  using namespace angle_units::operators;
  for (auto const &collectionTag : collectionTags_) {
    edm::Handle<std::vector<PSimHit>> hSimHits;
    event.getByLabel(collectionTag, hSimHits);

    for (auto const &simHit : *hSimHits) {
      DetId id(0);

      // --- Use only hits compatible with the in-time bunch-crossing
      if (simHit.tof() < 0 || simHit.tof() > 25.)
        continue;

      id = simHit.detUnitId();

      if (id == DetId(0)) {
        edm::LogWarning(messageCategory_) << "Invalid DetId for the current simHit!";
        continue;
      }

      if (simHit.trackId() == 0) {
        continue;
      }

      returnValue.emplace_back(id, &simHit);

      // get an unique id: for BTL the detId is unique (one for each crystal), for ETL the detId is not enough
      // also row and column are needed. An unique number is created from detId, row, col
      // Get row and column
      const auto &position = simHit.localPosition();
      LocalPoint simscaled(convertMmToCm(position.x()), convertMmToCm(position.y()), convertMmToCm(position.z()));
      std::pair<uint8_t, uint8_t> pixel = geomTools_.pixelInModule(id, simscaled);
      // create the unique id
      uint64_t uniqueId = static_cast<uint64_t>(id.rawId()) << 32;
      uniqueId |= pixel.first << 16;
      uniqueId |= pixel.second;

      simTrackDetIdEnergyMap[simHit.trackId()][uniqueId] += simHit.energyLoss();
      m_detIdToTotalSimEnergy[uniqueId] += simHit.energyLoss();
      // --- Get the time of the first SIM hit in the cell
      if (simTrackDetIdTimeMap[simHit.trackId()][uniqueId] == 0. ||
          simHit.tof() < simTrackDetIdTimeMap[simHit.trackId()][uniqueId]) {
        simTrackDetIdTimeMap[simHit.trackId()][uniqueId] = simHit.tof();
      }

#ifdef PRINT_DEBUG
      IfLogDebug(DEBUG, messageCategory_)
          << "hitId " << id.rawId() << " from track " << simHit.trackId() << " in layer " << geomTools_.layer(id)
          << ", module " << geomTools_.module(id) << ", pixel ( " << (int)geomTools_.pixelInModule(id, simscaled).first
          << ", " << (int)geomTools_.pixelInModule(id, simscaled).second << " )\n global pos(cm) "
          << geomTools_.globalPosition(id, simscaled) << ", time(ns) " << simHit.tof() << ", energy(MeV) "
          << convertUnitsTo(0.001_MeV, simHit.energyLoss()) << std::endl;
#endif
    }  // end of loop over simHits
  }    // end of loop over InputTags
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR(MtdTruthAccumulator);
