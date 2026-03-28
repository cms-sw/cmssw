/**
 * @brief CaloTruthAccumulator creates simulation truth objects for calorimeter : SimCluster and CaloParticle
 * Creates collections :
 *   - CaloParticle (both in CaloParticle and SimCluster dataformat) : set of simhits created by a genParticle (and all its decay products etc)
 *   - SimCluster "boundary" : for every SimTrack crossing the tracker-calorimeter boundary, create a SimCluster object collecting all simhits from it and its decay products
 *   - SimCluster "legacy" : every SimTrack with simhits makes a SimCluster (depends on SimTrack persistence criteria)
 *   - SimCluster "merged" : takes "boundary" SimClusters and merges them according to some algorithm (to merge SimClusters that are not separable at reconstruction level)
 *  and RefVectors to map back to CaloParticle (and map "boundary" to "merged")
 */
#include <iterator>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <tuple>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "SimGeneral/MixingModule/interface/DecayGraph.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <CLHEP/Units/SystemOfUnits.h>
#include <fastjet/ClusterSequence.hh>
#include <boost/container/flat_map.hpp>

namespace {
  using Index_t = unsigned;
  using Barcode_t = int;

  /** Config for building one SimCluster collection */
  struct SimClusterConfig {
    SimClusterCollection outputClusters;
    edm::EDPutTokenT<SimClusterCollection> outputClusters_token;

    /// For the map back to "CaloParticle" in SimCluster dataformat
    SimClusterRefVector clustersToCaloParticleMap;
    edm::EDPutTokenT<SimClusterRefVector> clustersToCaloParticleMap_token;

    SimClusterConfig(edm::ProducesCollector &c, std::string tag)
        : outputClusters_token(c.produces<SimClusterCollection>(tag)),
          clustersToCaloParticleMap_token(c.produces<SimClusterRefVector>(tag)) {}

    void clearAndReleaseMemory() {
      // Using the clear-and-minimize idiom to ensure the vector memory is completely released after each event is processed
      SimClusterCollection().swap(outputClusters);
      SimClusterRefVector().swap(clustersToCaloParticleMap);
    }
  };

  /**
  * Base class for a builder of SimCluster that has callbacks both when a CaloParticle is ended or when a SimCluster (inside a CaloParticle) is started
  * Typically used by SimCluster merging algorithm, in the following way : 
  *  1) a CaloParticle is started
  *     2) nested inside CaloParticle, several SimCluster can be built
  *  3) when a CaloParticle is ended, the merging algorithm merges some SimCluster and adds some "merged" SimCluster to another collection
  */
  class SubClusterMergerBase {
  public:
    /// Called when a SimCluster is started (along with the index of the SimCluster in its output collection)
    void start_subcluster(DecayChain::edge_descriptor e, const DecayChain &g, std::size_t currentSubClusterIndex);
    /**
     * Called when a CaloParticle is finished.
     * @param subClustersBuilt is the list of SimCluster built such that the indices "currentSubClusterIndex" passed to "start_subcluster" match
     */
    void end_parentCluster(std::span<SimCluster> subClustersBuilt, std::size_t currentCaloParticleIndex);
  };

  /// Uses fastjet jet clustering algorithm to merge SimCluster (only those from the same CaloParticle)
  template <typename ClusterParentIndexRecorderT>
  class SimClusterMergerByFastJet : public SubClusterMergerBase {
  public:
    SimClusterMergerByFastJet(SimClusterCollection &clusters,
                              ClusterParentIndexRecorderT caloParticleParentIndexRecorder,
                              ClusterParentIndexRecorderT subClusterToMergedClusterParentIndexRecorder,
                              fastjet::JetDefinition const &jetDefinition)
        : clusters_(clusters),
          caloParticleParentIndexRecorder_(caloParticleParentIndexRecorder),
          subClusterToMergedClusterParentIndexRecorder_(subClusterToMergedClusterParentIndexRecorder),
          jetDefinition_(jetDefinition) {}

    void start_subcluster(DecayChain::edge_descriptor e, const DecayChain &g, std::size_t currentSubClusterIndex) {
      auto edge_property = get(edge_weight, g, e);
      SimTrack const &simtrack = *edge_property.simTrack;

      /* Build the particle 4-vector such that the energy is the energy of SimTrack at boundary,
       the momentum 3-vector points to the boundary position, and the mass is zero */
      auto energyAtBoundary = simtrack.getMomentumAtBoundary().E();
      auto momentum3D = energyAtBoundary * simtrack.getPositionAtBoundary().Vect().Unit();
      auto &jet = fjInputs_.emplace_back(momentum3D.X(), momentum3D.Y(), momentum3D.Z(), energyAtBoundary);
      jet.set_user_index(currentSubClusterIndex);  // Store the index in SimCLuster collection for later
    }

    void end_parentCluster(std::span<SimCluster> subClustersBuilt, std::size_t currentCaloParticleIndex) {
      // Clustering
      fastjet::ClusterSequence sequence(fjInputs_, jetDefinition_);
      auto jets = sequence.inclusive_jets();

      /// Map from index of subcluster to index of merged cluster
      boost::container::flat_map<std::size_t, std::size_t> mapToSub;
      mapToSub.reserve(fjInputs_.size());

      // Merging
      for (fastjet::PseudoJet const &jet : jets) {
        auto constituents = fastjet::sorted_by_E(jet.constituents());
        assert(!constituents.empty());
        assert(constituents[0].user_index() < static_cast<int>(subClustersBuilt.size()));
        for (fastjet::PseudoJet const &pseudoJet : constituents) {
          // Index of merged cluster is clusters_.size() since we will emplace it just below
          mapToSub[pseudoJet.user_index()] = clusters_.size();
        }

        clusters_.emplace_back(SimCluster::mergeHitsFromCollection(
            constituents | std::views::transform([&](fastjet::PseudoJet const &pseudoJet) {
              return subClustersBuilt[static_cast<std::size_t>(pseudoJet.user_index())];
            })));

        // Record CaloParticle index for the merged SimCluster
        caloParticleParentIndexRecorder_.recordParentClusterIndex(currentCaloParticleIndex);
      }

      for (fastjet::PseudoJet const &subCluster : fjInputs_) {
        // Important to keep fjInputs_ in sync with subcluster indices
        subClusterToMergedClusterParentIndexRecorder_.recordParentClusterIndex(mapToSub[subCluster.user_index()]);
      }

      fjInputs_.clear();
    }

  private:
    SimClusterCollection &clusters_;  ///< output merged clusters

    std::vector<fastjet::PseudoJet> fjInputs_;  ///< SubClusters to be merged when end_parentCluster is called
    ClusterParentIndexRecorderT caloParticleParentIndexRecorder_;  ///< build RefVector to CaloParticle
    /// build RefVector from "sub" cluster to "this" cluster collection
    ClusterParentIndexRecorderT subClusterToMergedClusterParentIndexRecorder_;
    fastjet::JetDefinition const &jetDefinition_;
  };

  class SimClusterMergerByFastJetConfig : public SimClusterConfig {
  public:
    SimClusterMergerByFastJetConfig(edm::ProducesCollector &c, std::string tag, const edm::ParameterSet &ps)
        : SimClusterConfig(c, tag),
          subClustersToMergedClusterMap_token(c.produces<SimClusterRefVector>(tag + "MapFromSubCluster")),
          jetClusteringRadius_(ps.getParameter<double>("jetClusteringRadius")),
          jetDefinition_(fastjet::antikt_algorithm, jetClusteringRadius_) {}

    void fillPSetDescription(edm::ParameterSetDescription &desc) {
      desc.add<double>("jetClusteringRadius", 0.05)->setComment("Distance parameter for clustering algorithm");
    }

    template <typename ClusterParentIndexRecorderT>
    auto getMerger(ClusterParentIndexRecorderT caloParticleParentIndexRecorder,
                   ClusterParentIndexRecorderT subClusterToMergedClusterParentIndexRecorder) {
      return SimClusterMergerByFastJet(outputClusters,
                                       caloParticleParentIndexRecorder,
                                       subClusterToMergedClusterParentIndexRecorder,
                                       jetDefinition_);
    }

    void clearAndReleaseMemory() /* override */ {
      SimClusterConfig::clearAndReleaseMemory();
      SimClusterRefVector().swap(subClustersToMergedClusterMap);
    }

    /// For the map from subCluster to merged SimCluster
    SimClusterRefVector subClustersToMergedClusterMap;
    edm::EDPutTokenT<SimClusterRefVector> subClustersToMergedClusterMap_token;

  private:
    double jetClusteringRadius_;
    const fastjet::JetDefinition jetDefinition_;
  };
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

  /**
 * @brief Iterate over all SimHits collections (vectors of PCaloHit) and build maps between tracks, DetId and simEnergies
 * @return nested map G4 Track ID -> DetId -> accumulated SimHit energy (over same track, in case of loops)
 * Side-effect : Also this->m_detIdToTotalSimEnergy is filled (map DetId-> accumulated sim energy), for normalization into fractions in finalizeEvent
 */
  template <class T>
  std::unordered_map<int, std::map<int, float>> fillSimTrackDetIdEnergyMap(const T &event,
                                                                           const edm::EventSetup &setup);

  const std::string messageCategory_;

  /// Map DetId-> accumulated sim energy, to keep track of cell normalizations
  std::unordered_map<Index_t, float> m_detIdToTotalSimEnergy;

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

  std::vector<edm::InputTag> collectionTags_;  ///< SimHits collections
  const edm::InputTag genParticleLabel_;
  /// Needed to add HepMC::GenVertex to SimVertex
  const edm::InputTag hepMCproductLabel_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESWatcher<CaloGeometryRecord> geomWatcher_;

  const double minEnergy_, maxPseudoRapidity_;  ///< CaloParticle creation criteria
  bool produceLegacySimCluster_, produceBoundaryAndMergedSimCluster_;
  const bool premixStage1_;

  CaloParticleCollection outputCaloParticles_;
  edm::EDPutTokenT<CaloParticleCollection> outputCaloParticles_token_;

  SimClusterConfig legacySimClusters_config_;    ///< Legacy SimCluster from every SimTrack with simhits
  SimClusterConfig boundarySimClusters_config_;  ///< SimClusters from each SimTrack crossing boundary
  /// SimCluster that are identical to CaloParticle (to make it easier on downstream code, only one dataformat for everything)
  SimClusterConfig caloParticleSimClusters_config_;
  /// SimCluster built by merging "boundary" SimCluster (coming from the same CaloParticle) that are very close together  (using jet clustering algorithm)
  SimClusterMergerByFastJetConfig simClusterMergerFastJet_config_;
  /// apply a function to all SimCluster config
  template <typename T>
  void applyToSimClusterConfig(T f) {
    if (produceLegacySimCluster_ && produceBoundaryAndMergedSimCluster_)
      std::apply([f](auto &&...args) { (f(args), ...); },
                 std::tie(legacySimClusters_config_,
                          boundarySimClusters_config_,
                          caloParticleSimClusters_config_,
                          simClusterMergerFastJet_config_));
    else if (produceLegacySimCluster_)
      f(legacySimClusters_config_);
    else if (produceBoundaryAndMergedSimCluster_)
      std::apply(
          [f](auto &&...args) { (f(args), ...); },
          std::tie(boundarySimClusters_config_, caloParticleSimClusters_config_, simClusterMergerFastJet_config_));
  }

  /// To build the RefVector from SimCluster to "CaloParticle" SimCluster collection
  edm::RefProd<SimClusterCollection> caloParticles_refProd_;
  /// To build the RefVector from "merged" SimCluster to "boundary" SimCluster collection
  edm::RefProd<SimClusterCollection> boundarySimCluster_refProd_;

  const HGCalTopology *hgtopo_[3] = {nullptr, nullptr, nullptr};
  const HGCalDDDConstants *hgddd_[3] = {nullptr, nullptr, nullptr};
  const HcalDDDRecConstants *hcddd_ = nullptr;
  // geometry type (0 pre-TDR; 1 TDR)
  int geometryType_;
  const bool doHGCAL;
};

/* Graph utility functions */

namespace {
  /// Helper class to access hits, energies and time
  class VisitorHelper {
  public:
    VisitorHelper(std::unordered_map<int, std::map<int, float>> const &simTrackDetIdEnergyMap,
                  std::unordered_map<uint32_t, float> const &vertex_time_map)
        : simTrackDetIdEnergyMap_(simTrackDetIdEnergyMap), vertex_time_map_(vertex_time_map) {}

    // to check if SimTrack has simHits : use DecayGraph EdgeProperty

    /** Given a track G4 ID, give map DetId->accumulated SimHit energy  */
    std::map<int, float> const &hits_and_energies(unsigned int trackIdx) const {
      return simTrackDetIdEnergyMap_.at(trackIdx);
    }

    float getVertexTime(uint32_t simVertexId) { return vertex_time_map_.at(simVertexId); }

  private:
    /// nested map G4 Track ID -> DetId -> accumulated SimHit energy (over same track, in case of loops)
    std::unordered_map<int, std::map<int, float>> const &simTrackDetIdEnergyMap_;
    std::unordered_map<uint32_t, float> const &vertex_time_map_;  ///< Map SimVertex index to sim vertex time
  };

  /** 
   * Accumlates SimHits energies and DetId as each graph edge (=SimTrack) is visited
   * @tparam SimClassNameT either SimCluster or CaloParticle
   */
  template <typename SimClassNameT>
  class ClusterEnergyAccumulator {
  public:
    ClusterEnergyAccumulator(VisitorHelper const &helper) : helper_(helper) {}

    template <class Edge, class Graph>
    void accumulate_edge_in_cluster(Edge e, const Graph &g) {
      auto edge_property = get(edge_weight, g, e);
      const SimTrack *edge_simTrack = edge_property.simTrack;

      if (!edge_simTrack) [[unlikely]]
        return;  // Should not happen
      auto trackIdx = edge_simTrack->trackId();
      if (edge_property.simHits != 0) {
        for (auto const &hit_and_energy : helper_.hits_and_energies(trackIdx)) {
          acc_energy[hit_and_energy.first] += hit_and_energy.second;
        }
      }
    }

    void doEndCluster(SimClassNameT &cluster) {
      for (auto const &hit_and_energy : acc_energy) {
        cluster.addRecHitAndFraction(hit_and_energy.first, hit_and_energy.second);
      }
      acc_energy.clear();
    }

  private:
    VisitorHelper const &helper_;
    std::unordered_map<uint32_t, float> acc_energy;  ///< Map DetId->simHit energies for the current cluster
  };

  /**
   * Visitor class for depth_first_search.
   *  - on a root edge, check the associated GenParticle. Check if it passes selection
   *     if it does not, discard 
   *     if it does, start a new CaloParticle (record edge)
   *  - on any edge within a CaloParticle, check SimCluster conditions. 
   *    If they pass, create a SimCluster. 
   * @tparam Selector_t lambda for selections to create a CaloParticle
   * @tparam SubClusterVisitorTuple std::tuple of visitors for creating SimCluster collections nested inside the CaloParticle
   */
  template <typename Selector_t, typename SubClusterVisitorTuple>
  class PrimaryVisitor : public boost::default_dfs_visitor {
  public:
    /**
   * @param caloParticles output collection
   * @param caloParticleSelector functor EdgeProperty -> bool to determine if a SimTrack should be promoted to CaloParticle
   * @param subClusterVisitors tuple of visitors for callbacks (use as std::make_tuple(SubClusterVisitor<SimCluster>( ...), SubClusterVisitor<...>(...), ....) )
   */
    PrimaryVisitor(VisitorHelper &helper,
                   std::vector<CaloParticle> &caloParticles,
                   Selector_t caloParticleSelector,
                   SubClusterVisitorTuple subClusterVisitors)
        : helper_(helper),
          caloParticleSelector_(caloParticleSelector),
          caloParticleAccumulator_(helper),
          caloParticles_(caloParticles),
          subClusterVisitors_(subClusterVisitors) {}

    void examine_edge(DecayChain::edge_descriptor e, const DecayChain &g) {
      auto edge_property = get(edge_weight, g, e);
      if (!insideCluster_ && caloParticleSelector_(edge_property)) {
        insideCluster_ = true;
        caloParticleEdge_ = e;
        // Create a new CaloParticle
        caloParticles_.emplace_back(*edge_property.simTrack);
        // For a CaloParticle the simTime is set at the vertex !
        caloParticles_.back().setSimTime(helper_.getVertexTime(edge_property.simTrack->vertIndex()));
        // This loops over all elements of subClusterVisitors_ tuple and calls begin_parentCluster on them (compile-time loop expansion)
        std::apply([&](auto &...x) { (..., x.begin_parentCluster(e, g)); }, subClusterVisitors_);
      }

      if (insideCluster_) {
        caloParticleAccumulator_.accumulate_edge_in_cluster(e, g);  // accumulate simhits energies

        std::apply([&](auto &...x) { (..., x.examine_edge(e, g, caloParticles_.size() - 1)); }, subClusterVisitors_);
      }
    }

    void finish_edge(DecayChain::edge_descriptor e, const DecayChain &g) {
      if (insideCluster_) {
        std::apply([&](auto &...x) { (..., x.finish_edge(e, g)); }, subClusterVisitors_);
      }
      if (insideCluster_ && e == caloParticleEdge_) {
        insideCluster_ = false;
        caloParticleAccumulator_.doEndCluster(caloParticles_.back());
        std::apply([&](auto &...x) { (..., x.end_parentCluster(caloParticles_.size() - 1)); }, subClusterVisitors_);
      }
    }

  private:
    VisitorHelper &helper_;

    Selector_t caloParticleSelector_;
    ClusterEnergyAccumulator<CaloParticle> caloParticleAccumulator_;
    std::vector<CaloParticle> &caloParticles_;  ///< Output collection

    SubClusterVisitorTuple subClusterVisitors_;  ///< std::tuple of SubClusterVisitor

    bool insideCluster_{false};  ///< are we inside a CaloParticle (during DFS algorithm)
    /// Keep track of CaloParticle root edge to flag end of cluster in DFS
    DecayChain::edge_descriptor caloParticleEdge_;
  };

  /** 
   * Fills RefVector to keep mapping from SimCluster to another collection (typically CaloParticle).
   * @tparam ParentClusterCollectionT type of the collection to refer to (eg: std::vector<SimCluster> or std::vector<CaloParticle>)
   */
  template <typename ParentClusterCollectionT>
  class ClusterParentIndexRecorder {
  public:
    /** The RefProd is to fill the RefVector with edm::Ref (build it with getRefBeforePut) */
    ClusterParentIndexRecorder(edm::RefVector<ParentClusterCollectionT> &refVector,
                               edm::RefProd<ParentClusterCollectionT> const &refProd)
        : refVector_(refVector), refProd_(refProd) {}

    void recordParentClusterIndex(std::size_t parentClusterIndex) {
      refVector_.push_back(edm::Ref<ParentClusterCollectionT>(refProd_, parentClusterIndex));
    }

  private:
    edm::RefVector<ParentClusterCollectionT> &refVector_;  ///< output RefVector
    /// Need the RefProd to build the edm::Ref to insert into RefVector
    edm::RefProd<ParentClusterCollectionT> const &refProd_;
  };

  /** Base class for a graph visitor building SubCluster (ie SimCluster that are created inside a CaloParticle) */
  class SubClusterVisitorBase {
  public:
    /// Called during DFS at each edge(=SimTrack) inside a CaloParticle
    void examine_edge(DecayChain::edge_descriptor e, const DecayChain &g, std::size_t currentCaloParticleIndex) {}
    void finish_edge(DecayChain::edge_descriptor e, const DecayChain &g) {}

    /// Called when a "parent" cluster (ie a CaloParticle) is started
    void begin_parentCluster(DecayChain::edge_descriptor e, const DecayChain &g) {}
    /// Called when a "parent" cluster (ie a CaloParticle) is ended, along with the CaloParticle index into its own collection (for building edm::Ref)
    void end_parentCluster(std::size_t currentCaloParticleIndex) {}
  };

  /** 
   * Visitor that creates cluster with reference to another "parent" cluster collection, like SimCluster inside CaloParticle
   * Does not make "nested sub-clusters"
   * @tparam SubClusterT typically SimCluster
   * @tparam ClusterParentIndexRecorderT type of the object in charge of building the RefVector to parent (@see ClusterParentIndexRecorder)
   * @tparam Selector_t lambda for criteria for creating a subcluster
   */
  template <typename SubClusterT,
            typename ClusterParentIndexRecorderT,
            typename Selector_t,
            typename SubClusterMergerTupleT = std::tuple<>>
  class SubClusterVisitor : public SubClusterVisitorBase {
  public:
    SubClusterVisitor(std::vector<SubClusterT> &clusters,
                      ClusterParentIndexRecorderT parentIndexRecorder,
                      VisitorHelper const &helper,
                      Selector_t selector,
                      SubClusterMergerTupleT subClusterMergers = SubClusterMergerTupleT())
        : clusters_(clusters),
          accumulator(helper),
          indexRecorder(parentIndexRecorder),
          selector_(selector),
          subClusterMergers_(subClusterMergers) {}

    void examine_edge(DecayChain::edge_descriptor e, const DecayChain &g, std::size_t currentCaloParticleIndex) {
      if (!insideCluster_ && selector_(get(edge_weight, g, e))) {
        insideCluster_ = true;
        clusterRootEdge_ = e;

        // Create a new cluster
        auto edge_property = get(edge_weight, g, e);
        clusters_.emplace_back(*edge_property.simTrack);
        indexRecorder.recordParentClusterIndex(currentCaloParticleIndex);

        std::apply([&](auto &...x) { (..., x.start_subcluster(e, g, clusters_.size() - 1)); }, subClusterMergers_);
      }

      if (insideCluster_) {
        accumulator.accumulate_edge_in_cluster(e, g);
      }
    }

    void finish_edge(DecayChain::edge_descriptor e, const DecayChain &g) {
      if (insideCluster_ && e == clusterRootEdge_) {  // we backtracked to the starting edge
        insideCluster_ = false;
        accumulator.doEndCluster(clusters_.back());
      }
    }

    void end_parentCluster(std::size_t currentCaloParticleIndex) {
      std::apply([&](auto &...x) { (..., x.end_parentCluster(clusters_, currentCaloParticleIndex)); },
                 subClusterMergers_);
    }

  private:
    std::vector<SubClusterT> &clusters_;                ///< output
    ClusterEnergyAccumulator<SubClusterT> accumulator;  ///< keep track of simhits
    ClusterParentIndexRecorderT indexRecorder;          ///< build RefVector to CaloParticle
    Selector_t selector_;                               ///< lambda for criteria for creating a subcluster
    bool insideCluster_{false};                         ///< is the current DFS algo position inside a subcluster
    DecayChain::edge_descriptor clusterRootEdge_;       ///< root edge of the subcluster

    /// tuple of mergers that get called at the end of a cluster to possibly do some merging
    SubClusterMergerTupleT subClusterMergers_;
  };

  /**Creates SimCluster for every SimTrack with simHits. One SimCluster only includes the simHits from the SimTrack it was made of (depends on SimTrack saving criteria) */
  template <typename SubClusterT, typename ClusterParentIndexRecorderT>
  class LegacySimClusterVisitor : public SubClusterVisitorBase {
  public:
    LegacySimClusterVisitor(std::vector<SubClusterT> &clusters,
                            ClusterParentIndexRecorderT parentIndexRecorder,
                            VisitorHelper const &helper)
        : clusters_(clusters), accumulator(helper), indexRecorder(parentIndexRecorder) {}

    void examine_edge(DecayChain::edge_descriptor e, const DecayChain &g, std::size_t currentCaloParticleIndex) {
      auto edge_property = get(edge_weight, g, e);
      if (edge_property.simHits != 0) {
        // Create a new cluster
        clusters_.emplace_back(*edge_property.simTrack);
        indexRecorder.recordParentClusterIndex(currentCaloParticleIndex);
        accumulator.accumulate_edge_in_cluster(e, g);
        accumulator.doEndCluster(clusters_.back());
      }
    }
    void finish_edge(DecayChain::edge_descriptor e, const DecayChain &g) {}

  private:
    std::vector<SubClusterT> &clusters_;
    ClusterEnergyAccumulator<SubClusterT> accumulator;
    ClusterParentIndexRecorderT indexRecorder;
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
      produceLegacySimCluster_(config.getParameter<bool>("produceLegacySimCluster")),
      produceBoundaryAndMergedSimCluster_(config.getParameter<bool>("produceBoundaryAndMergedSimCluster")),
      premixStage1_(config.getParameter<bool>("premixStage1")),
      outputCaloParticles_token_(producesCollector.produces<CaloParticleCollection>("MergedCaloTruth")),
      legacySimClusters_config_(producesCollector, "MergedCaloTruth"),
      boundarySimClusters_config_(producesCollector, "MergedCaloTruthBoundaryTrackSimCluster"),
      caloParticleSimClusters_config_(producesCollector, "MergedCaloTruthCaloParticle"),
      simClusterMergerFastJet_config_(producesCollector,
                                      "MergedCaloTruthMergedSimCluster",
                                      config.getParameter<edm::ParameterSet>("simClusterMergerConfig")),
      geometryType_(-1),
      doHGCAL(config.getParameter<bool>("doHGCAL")) {
  if (premixStage1_) {
    producesCollector.produces<std::vector<std::pair<unsigned int, float>>>(
        "MergedCaloTruth");  // (DetId, total simhit energy) pairs
  }

  iC.consumes<std::vector<SimTrack>>(simTrackLabel_);
  iC.consumes<std::vector<SimVertex>>(simVertexLabel_);
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
  // Seems const_cast is necessary here
  caloParticles_refProd_ = const_cast<edm::Event &>(event).getRefBeforePut<SimClusterCollection>(
      caloParticleSimClusters_config_.outputClusters_token);
  if (produceBoundaryAndMergedSimCluster_)
    boundarySimCluster_refProd_ = const_cast<edm::Event &>(event).getRefBeforePut<SimClusterCollection>(
        boundarySimClusters_config_.outputClusters_token);
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

namespace {
  /** Normalize the energies in the SimCluster/CaloParticle collection, from absolute SimHit energies to fraction of total simHits energies */
  template <typename SimCaloCollection>
  void normalizeCollection(SimCaloCollection &simClusters,
                           std::unordered_map<Index_t, float> const &detIdToTotalSimEnergy) {
    for (auto &sc : simClusters) {
      auto hitsAndEnergies = sc.hits_and_fractions();
      sc.clearHitsAndFractions();
      // Note : addHitEnergy is actually never used, so we do not clear and refill for it
      for (auto &hAndE : hitsAndEnergies) {
        const float totalenergy = detIdToTotalSimEnergy.at(hAndE.first);
        float fraction = 0.;
        if (totalenergy > 0)
          fraction = hAndE.second / totalenergy;
        else
          edm::LogWarning("CaloTruthAccumulator")
              << "TotalSimEnergy for hit " << hAndE.first << " is 0! The fraction for this hit cannot be computed.";
        sc.addRecHitAndFraction(hAndE.first, fraction);
      }
    }
  }
};  // namespace

void CaloTruthAccumulator::finalizeEvent(edm::Event &event, edm::EventSetup const &setup) {
  edm::LogInfo(messageCategory_) << "Adding " << legacySimClusters_config_.outputClusters.size()
                                 << " legacy SimClusters and " << outputCaloParticles_.size()
                                 << " CaloParticles to the event.";

  // We need to normalize the hits and energies into hits and fractions (since
  // we have looped over all pileup events)
  // For premixing stage1 we keep the energies, they will be normalized to
  // fractions in stage2 (in PreMixingCaloParticleWorker.cc)

  if (premixStage1_) {
    auto totalEnergies = std::make_unique<std::vector<std::pair<unsigned int, float>>>();
    totalEnergies->reserve(m_detIdToTotalSimEnergy.size());
    std::copy(m_detIdToTotalSimEnergy.begin(), m_detIdToTotalSimEnergy.end(), std::back_inserter(*totalEnergies));
    std::sort(totalEnergies->begin(), totalEnergies->end());
    event.put(std::move(totalEnergies), "MergedCaloTruth");
  } else {
    applyToSimClusterConfig(
        [this](auto &config) { normalizeCollection(config.outputClusters, m_detIdToTotalSimEnergy); });
  }

  if (produceLegacySimCluster_) {
    // fill the calo particles with their ref to SimCluster
    auto legacySimClustersRefProd = event.getRefBeforePut(legacySimClusters_config_.outputClusters_token);
    for (unsigned i = 0; i < legacySimClusters_config_.outputClusters.size(); ++i) {
      // get the key of the edm::Ref<CaloParticle>
      auto &cp = outputCaloParticles_[legacySimClusters_config_.clustersToCaloParticleMap[i].key()];
      cp.addSimCluster(edm::Ref<SimClusterCollection>(legacySimClustersRefProd, i));
    }
  }

  event.emplace(outputCaloParticles_token_, std::move(outputCaloParticles_));
  // puts the vector into "empty" determined state (instead of "moved-from" undetermined state)
  // also uses clear-and-minimize idiom to ensure the memory is released
  CaloParticleCollection().swap(outputCaloParticles_);

  applyToSimClusterConfig([&event](auto &config) {
    event.emplace(config.outputClusters_token, std::move(config.outputClusters));
    event.emplace(config.clustersToCaloParticleMap_token, std::move(config.clustersToCaloParticleMap));
    // puts the vector into "empty" determined state (instead of "moved-from" undetermined state)
    // and ensure the memory is released (should already be released due to the move, but make sure)
    config.clearAndReleaseMemory();
  });
  if (produceBoundaryAndMergedSimCluster_)
    event.emplace(simClusterMergerFastJet_config_.subClustersToMergedClusterMap_token,
                  std::move(simClusterMergerFastJet_config_.subClustersToMergedClusterMap));

  std::unordered_map<Index_t, float>().swap(m_detIdToTotalSimEnergy);  // clear-and-minimize idiom
}

template <class T>
void CaloTruthAccumulator::accumulateEvent(const T &event,
                                           const edm::EventSetup &setup,
                                           const edm::Handle<edm::HepMCProduct> &hepMCproduct) {
  edm::Handle<std::vector<reco::GenParticle>> hGenParticles;
  edm::Handle<std::vector<int>> hGenParticleIndices;
  edm::Handle<std::vector<SimTrack>> hSimTracks;
  edm::Handle<std::vector<SimVertex>> hSimVertices;
  // We must always use getByLabel (event can be PileupEventPrincipal which does not have the get, getByToken, etc functions)
  event.getByLabel(simTrackLabel_, hSimTracks);
  event.getByLabel(simVertexLabel_, hSimVertices);

  event.getByLabel(genParticleLabel_, hGenParticles);
  event.getByLabel(genParticleLabel_, hGenParticleIndices);

  std::unordered_map<int, std::map<int, float>> simTrackDetIdEnergyMap = fillSimTrackDetIdEnergyMap(event, setup);

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

  std::unordered_map<uint32_t, float> vertex_time_map;
  for (uint32_t i = 0; i < vertices.size(); i++) {
    // Geant4 time is in seconds, convert to ns (CLHEP::s = 1e9)
    vertex_time_map[i] = vertices[i].position().t() * CLHEP::s;
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

  VisitorHelper visitorHelper(simTrackDetIdEnergyMap, vertex_time_map);

  auto passGenPartSelections_lambda = [&](EdgeProperty const &edge_property) {
    return (edge_property.cumulative_simHits != 0 and !edge_property.simTrack->noGenpart() and
            edge_property.simTrack->momentum().E() > minEnergy_ and
            std::abs(edge_property.simTrack->momentum().Eta()) < maxPseudoRapidity_);
  };

  auto makeCaloParticleSimCluster = [&]() {  // "CaloParticle" SimCluster
    return SubClusterVisitor(
        caloParticleSimClusters_config_.outputClusters,
        // Trivial 1-1 mapping, kept for convenience
        ClusterParentIndexRecorder(caloParticleSimClusters_config_.clustersToCaloParticleMap, caloParticles_refProd_),
        visitorHelper,
        [&](EdgeProperty const &edge_property) -> bool {
          // Create a SimCluster for every CaloParticle (duplicates the CaloParticle for convenience of use, to have one single dataformat)
          return true;
        });
  };
  auto makeLegacySimCluster = [&]() {  // "legacy" SimCluster (from every SimTrack with simhits)
    return LegacySimClusterVisitor(
        legacySimClusters_config_.outputClusters,
        ClusterParentIndexRecorder(legacySimClusters_config_.clustersToCaloParticleMap, caloParticles_refProd_),
        visitorHelper);
  };
  auto makeBoundarySimCluster = [&]() {  // "boundary" SimCluster (from every SimTrack crossing boundary)
    return SubClusterVisitor(
        boundarySimClusters_config_.outputClusters,
        ClusterParentIndexRecorder(boundarySimClusters_config_.clustersToCaloParticleMap, caloParticles_refProd_),
        visitorHelper,
        [&](EdgeProperty const &edge_property) -> bool {
          // Create SimCluster from every SimTrack crossing boundary (and which has simhits either itself as in its descendants), and that is inside a CaloParticle
          return edge_property.cumulative_simHits != 0 && edge_property.simTrack->crossedBoundary();
        },
        // "merging" subcluster configuration
        std::make_tuple(simClusterMergerFastJet_config_.getMerger(
            ClusterParentIndexRecorder(simClusterMergerFastJet_config_.clustersToCaloParticleMap,
                                       caloParticles_refProd_),
            ClusterParentIndexRecorder(simClusterMergerFastJet_config_.subClustersToMergedClusterMap,
                                       boundarySimCluster_refProd_))));
  };

  if (produceLegacySimCluster_ && produceBoundaryAndMergedSimCluster_) {
    // Do the graph search for all 3 vistors at the same time
    //  "visitor()" is a Boost BGL named parameter
    depth_first_search(
        decay,
        visitor(PrimaryVisitor(
            visitorHelper,
            outputCaloParticles_,
            passGenPartSelections_lambda,
            std::make_tuple(makeCaloParticleSimCluster(), makeLegacySimCluster(), makeBoundarySimCluster()))));
  } else if (produceLegacySimCluster_) {
    depth_first_search(decay,
                       visitor(PrimaryVisitor(visitorHelper,
                                              outputCaloParticles_,
                                              passGenPartSelections_lambda,
                                              std::make_tuple(makeLegacySimCluster()))));
  } else if (produceBoundaryAndMergedSimCluster_) {
    depth_first_search(
        decay,
        visitor(PrimaryVisitor(visitorHelper,
                               outputCaloParticles_,
                               passGenPartSelections_lambda,
                               std::make_tuple(makeCaloParticleSimCluster(), makeBoundarySimCluster()))));
  } else
    depth_first_search(
        decay,
        visitor(PrimaryVisitor(visitorHelper, outputCaloParticles_, passGenPartSelections_lambda, std::make_tuple())));

#if DEBUG
  boost::write_graphviz(std::cout,
                        decay,
                        make_label_writer(make_transform_value_property_map(&graphviz_vertex, get(vertex_name, decay))),
                        make_label_writer(make_transform_value_property_map(&graphviz_edge, get(edge_weight, decay))));
#endif
}

template <class T>
std::unordered_map<int, std::map<int, float>> CaloTruthAccumulator::fillSimTrackDetIdEnergyMap(
    const T &event, const edm::EventSetup &setup) {
  std::unordered_map<int, std::map<int, float>> simTrackDetIdEnergyMap;
  for (auto const &collectionTag : collectionTags_) {
    edm::Handle<std::vector<PCaloHit>> hSimHits;
    const bool isHcal = (collectionTag.instance().find("HcalHits") != std::string::npos);
    const bool isHGCal = (collectionTag.instance().find("HGCHits") != std::string::npos);
    event.getByLabel(collectionTag, hSimHits);

    for (auto const &simHit : *hSimHits) {
      DetId id(0);

      //Relabel as necessary for HGCAL
      if (isHGCal) {
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

      simTrackDetIdEnergyMap[simHit.geantTrackId()][id.rawId()] += simHit.energy();
      m_detIdToTotalSimEnergy[id.rawId()] += simHit.energy();
    }
  }  // end of loop over InputTags
  return simTrackDetIdEnergyMap;
}

// Register with the framework
DEFINE_DIGI_ACCUMULATOR(CaloTruthAccumulator);
