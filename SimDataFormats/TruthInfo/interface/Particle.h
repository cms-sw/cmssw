// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef SimDataFormats_TruthInfo_interface_Particle_h
#define SimDataFormats_TruthInfo_interface_Particle_h

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/Checkpoint.h"
#include "SimDataFormats/TruthInfo/interface/ParticleData.h"

namespace truth {

  class Graph;
  class Vertex;

  // A lightweight, copyable view of one logical particle. It stores only a graph
  // pointer and an id; all accessors read through to the owning Graph (the heavy
  // method bodies live in Graph.cc).
  class Particle {
  public:
    Particle() = default;
    Particle(Graph const* graph, uint32_t id) : graph_(graph), id_(id) {}

    [[nodiscard]] bool valid() const { return graph_ != nullptr; }
    [[nodiscard]] Graph const* graph() const { return graph_; }
    [[nodiscard]] uint32_t id() const { return id_; }

    [[nodiscard]] const ParticleData& data() const;

    [[nodiscard]] bool hasGen() const;
    [[nodiscard]] bool hasSim() const;
    [[nodiscard]] int32_t pdgId() const;
    [[nodiscard]] int16_t status() const;
    [[nodiscard]] uint16_t statusFlags() const;
    [[nodiscard]] uint64_t eventId() const;
    [[nodiscard]] int32_t genEvent() const;
    [[nodiscard]] bool backscattered() const;
    [[nodiscard]] const math::XYZTLorentzVectorD& momentum() const;
    [[nodiscard]] const math::XYZTLorentzVectorD& p4() const { return momentum(); }  // alias
    // Three times the electric charge, as HepPDT reports it: an electron gives -3. The
    // name says so, because a charge of -3 read as the charge itself is a factor 3.
    [[nodiscard]] int threeCharge() const;

    [[nodiscard]] std::span<const Checkpoint> checkpoints() const;
    [[nodiscard]] bool hasCheckpoints() const;
    [[nodiscard]] std::optional<Checkpoint> checkpoint(uint32_t checkpointId) const;

    [[nodiscard]] bool isRoot() const;
    [[nodiscard]] bool isLeaf() const;

    [[nodiscard]] std::vector<Vertex> productionVertices() const;
    [[nodiscard]] std::vector<Vertex> decayVertices() const;

    // These four build a vector on every call, and ancestors() and descendants() walk the
    // whole subgraph to do it. In a loop over particles use forEachChildId and
    // forEachParentId below, or the CSR spans of Graph, which allocate nothing.
    [[nodiscard]] std::vector<Particle> parents() const;
    [[nodiscard]] std::vector<Particle> children() const;

    [[nodiscard]] std::vector<Particle> ancestors() const;
    [[nodiscard]] std::vector<Particle> descendants() const;

    // How many particles sit above this one, which is ancestors().size() without
    // building the list.
    [[nodiscard]] uint32_t ancestorCount() const;

    // The ids of the particles this one decays into, and of those it comes from, passed
    // to the callable one at a time. An id can repeat when two vertices share a particle.
    // Defined in Graph.h, which a caller has to include.
    template <typename F>
    void forEachChildId(F&& visit) const;
    template <typename F>
    void forEachParentId(F&& visit) const;

    [[nodiscard]] bool hasAncestorPdgId(int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstAncestorWithPdgId(int pdgId) const;

    // The last copy of a radiating chain: follow the one same-species child through the
    // one decay vertex until the species changes, the particle is stable, or the step is
    // ambiguous. What a particle decays into is read from the copy this returns, so a
    // child lookup on a radiating particle starts here.
    [[nodiscard]] Particle lastCopy() const;

    // The first child of this particle with exactly this signed pdgId, the downward twin
    // of firstAncestorWithPdgId. A radiating particle carries its decay products on its
    // last copy, so ask lastCopy() first where that matters.
    [[nodiscard]] std::optional<Particle> firstChildWithPdgId(int pdgId) const;

    // The other particles produced where this one was produced, each once. This is what
    // recoils against the particle: the VBF tagging quarks, the single-top partner, the
    // vector boson made with a Higgs.
    [[nodiscard]] std::vector<Particle> productionSiblings() const;
    [[nodiscard]] std::optional<Particle> firstCommonAncestor(Particle other) const;

    [[nodiscard]] bool operator==(Particle const& other) const { return graph_ == other.graph_ && id_ == other.id_; }
    [[nodiscard]] bool operator!=(Particle const& other) const { return !(*this == other); }

  private:
    Graph const* graph_ = nullptr;
    uint32_t id_ = 0;
  };

}  // namespace truth

#endif
