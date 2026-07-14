// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Author: Felice Pantaleo - CERN
// Date: 03/2026
// A compact, read-only graph representation of the truth information in an event.
// The graph is built in the TruthGraphProducer module, which also fills the node metadata and associations.
// The graph is intended to be a common data format for various use cases (e.g. validation, analysis, visualization).

#ifndef SimDataFormats_TruthInfo_interface_TruthGraph_h
#define SimDataFormats_TruthInfo_interface_TruthGraph_h

#include <cstdint>
#include <span>
#include <vector>

class TruthGraph {
public:
  enum class NodeKind : uint8_t {
    GenEvent = 0,
    GenVertex = 1,
    GenParticle = 2,
    SimVertex = 3,
    SimTrack = 4,
  };

  // Edge categories (for visualization / filtering)
  enum class EdgeKind : uint8_t {
    Gen = 0,       // within GEN realm
    Sim = 1,       // within SIM realm
    GenToSim = 2,  // realm boundary GEN -> SIM
    SimToGen = 3   // reserved (we don't produce these now)
  };

  struct NodeRef {
    NodeKind kind = NodeKind::GenParticle;
    int64_t key = 0;  // GenParticle: index; SimTrack: trackId; SimVertex: index; GenVertex: barcode/index
  };

  TruthGraph() = default;

  // --- Storage accessors --------------------------------------------------
  // The graph is built once (the producer fills the vectors via the non-const
  // accessors) and then read many times (consumers use the const accessors and
  // the node helpers below). The CSR invariants are: offsets().size() == nNodes+1,
  // edges().size() == edgeKind().size() == nEdges.
  [[nodiscard]] std::vector<uint32_t> const& offsets() const { return offsets_; }
  [[nodiscard]] std::vector<uint32_t>& offsets() { return offsets_; }

  [[nodiscard]] std::vector<uint32_t> const& edges() const { return edges_; }
  [[nodiscard]] std::vector<uint32_t>& edges() { return edges_; }

  [[nodiscard]] std::vector<uint8_t> const& edgeKind() const { return edgeKind_; }
  [[nodiscard]] std::vector<uint8_t>& edgeKind() { return edgeKind_; }

  [[nodiscard]] std::vector<NodeRef> const& nodes() const { return nodes_; }
  [[nodiscard]] std::vector<NodeRef>& nodes() { return nodes_; }

  [[nodiscard]] std::vector<int32_t> const& pdgId() const { return pdgId_; }
  [[nodiscard]] std::vector<int32_t>& pdgId() { return pdgId_; }

  [[nodiscard]] std::vector<int16_t> const& status() const { return status_; }
  [[nodiscard]] std::vector<int16_t>& status() { return status_; }

  [[nodiscard]] std::vector<uint16_t> const& statusFlags() const { return statusFlags_; }
  [[nodiscard]] std::vector<uint16_t>& statusFlags() { return statusFlags_; }

  [[nodiscard]] std::vector<uint64_t> const& eventId() const { return eventId_; }
  [[nodiscard]] std::vector<uint64_t>& eventId() { return eventId_; }

  [[nodiscard]] std::vector<int32_t> const& genEventOfNode() const { return genEventOfNode_; }
  [[nodiscard]] std::vector<int32_t>& genEventOfNode() { return genEventOfNode_; }

  [[nodiscard]] std::vector<uint16_t> const& simVertexProcessType() const { return simVertexProcessType_; }
  [[nodiscard]] std::vector<uint16_t>& simVertexProcessType() { return simVertexProcessType_; }

  [[nodiscard]] std::vector<uint8_t> const& simTrackBackscattered() const { return simTrackBackscattered_; }
  [[nodiscard]] std::vector<uint8_t>& simTrackBackscattered() { return simTrackBackscattered_; }

  [[nodiscard]] std::vector<int32_t> const& simTrackToGen() const { return simTrackToGen_; }
  [[nodiscard]] std::vector<int32_t>& simTrackToGen() { return simTrackToGen_; }

  [[nodiscard]] std::vector<int32_t> const& simTrackToVtx() const { return simTrackToVtx_; }
  [[nodiscard]] std::vector<int32_t>& simTrackToVtx() { return simTrackToVtx_; }

  [[nodiscard]] std::vector<int32_t> const& simVtxToGen() const { return simVtxToGen_; }
  [[nodiscard]] std::vector<int32_t>& simVtxToGen() { return simVtxToGen_; }

  // --- Node / edge helpers ------------------------------------------------
  uint32_t nNodes() const { return static_cast<uint32_t>(nodes_.size()); }
  uint32_t nEdges() const { return static_cast<uint32_t>(edges_.size()); }

  uint32_t edgeBegin(uint32_t nodeId) const { return offsets_.at(nodeId); }
  uint32_t edgeEnd(uint32_t nodeId) const { return offsets_.at(nodeId + 1); }

  std::span<const uint32_t> children(uint32_t nodeId) const {
    const auto begin = edgeBegin(nodeId);
    const auto end = edgeEnd(nodeId);
    return std::span<const uint32_t>(edges_.data() + begin, end - begin);
  }

  std::span<const uint8_t> childrenEdgeKinds(uint32_t nodeId) const {
    const auto begin = edgeBegin(nodeId);
    const auto end = edgeEnd(nodeId);
    return std::span<const uint8_t>(edgeKind_.data() + begin, end - begin);
  }

  const NodeRef& nodeRef(uint32_t nodeId) const { return nodes_.at(nodeId); }

  int32_t nodePdgId(uint32_t nodeId) const { return (nodeId < pdgId_.size()) ? pdgId_[nodeId] : 0; }

  int16_t nodeStatus(uint32_t nodeId) const { return (nodeId < status_.size()) ? status_[nodeId] : 0; }
  uint16_t nodeStatusFlags(uint32_t nodeId) const { return (nodeId < statusFlags_.size()) ? statusFlags_[nodeId] : 0; }
  uint64_t nodeEventId(uint32_t nodeId) const { return (nodeId < eventId_.size()) ? eventId_[nodeId] : 0ull; }

  bool nodeBackscattered(uint32_t nodeId) const {
    return (nodeId < simTrackBackscattered_.size()) && simTrackBackscattered_[nodeId] != 0;
  }

  uint16_t nodeProcessType(uint32_t nodeId) const {
    return (nodeId < simVertexProcessType_.size()) ? simVertexProcessType_[nodeId] : 0;
  }

  int32_t nodeSimTrackToGen(uint32_t nodeId) const {
    return (nodeId < simTrackToGen_.size()) ? simTrackToGen_[nodeId] : -1;
  }

  int32_t nodeSimTrackToVtx(uint32_t nodeId) const {
    return (nodeId < simTrackToVtx_.size()) ? simTrackToVtx_[nodeId] : -1;
  }

  int32_t nodeSimVtxToGen(uint32_t nodeId) const { return (nodeId < simVtxToGen_.size()) ? simVtxToGen_[nodeId] : -1; }

  bool isConsistent() const;

private:
  // CSR out-edges: offsets_.size() == nNodes+1, edges_/edgeKind_.size() == nEdges.
  std::vector<uint32_t> offsets_;
  std::vector<uint32_t> edges_;
  std::vector<uint8_t> edgeKind_;  // stores TruthGraph::EdgeKind as uint8_t

  // Node metadata: nodes_.size() == nNodes
  std::vector<NodeRef> nodes_;

  // Cached payload (optional)
  std::vector<int32_t> pdgId_;         // 0 if not applicable
  std::vector<int16_t> status_;        // 0 if not applicable
  std::vector<uint16_t> statusFlags_;  // packed reco::GenStatusFlags, 0 if not available
  std::vector<uint64_t> eventId_;      // packed EncodedEventId for SIM nodes; 0 for GEN nodes

  std::vector<int32_t> genEventOfNode_;  // -1 for SIM; for GEN nodes = component id

  // Geant4 process *subtype* (G4VProcess::GetProcessSubType()) of the creator
  // process of the SimVertex's outgoing particles - the physical reason the vertex
  // exists (e.g. 2 = ionisation/delta-ray, 3 = bremsstrahlung, 14 = pair conversion,
  // 121 = hadronic inelastic, 201 = decay). 0 for primaries and non-SimVertex nodes.
  std::vector<uint16_t> simVertexProcessType_;

  // 1 for SimTrack nodes Geant4 flagged as back-scattered (SimTrack::isFromBack-
  // Scattering, the Tracker<->CALO inward albedo); 0 otherwise / non-SimTrack nodes.
  std::vector<uint8_t> simTrackBackscattered_;

  // Associations (nodeId -> nodeId). Only meaningful for SimTrack nodes.
  // -1 means "no association".
  std::vector<int32_t> simTrackToGen_;  // SimTrack nodeId -> GenParticle nodeId
  std::vector<int32_t> simTrackToVtx_;  // SimTrack nodeId -> SimVertex nodeId

  // SimVertex nodeId -> GenVertex nodeId provenance association, -1 if none.
  // Derived from primary SimTracks: a SimTrack's production SimVertex corresponds
  // to the production GenVertex of its associated GenParticle. Only meaningful for
  // SimVertex nodes.
  std::vector<int32_t> simVtxToGen_;
};

#endif
