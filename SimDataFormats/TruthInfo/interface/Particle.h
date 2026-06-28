// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

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

    [[nodiscard]] std::span<const Checkpoint> checkpoints() const;
    [[nodiscard]] bool hasCheckpoints() const;
    [[nodiscard]] std::optional<Checkpoint> checkpoint(uint32_t checkpointId) const;

    [[nodiscard]] bool isRoot() const;
    [[nodiscard]] bool isLeaf() const;

    [[nodiscard]] std::vector<Vertex> productionVertices() const;
    [[nodiscard]] std::vector<Vertex> decayVertices() const;

    [[nodiscard]] std::vector<Particle> parents() const;
    [[nodiscard]] std::vector<Particle> children() const;

    [[nodiscard]] std::vector<Particle> ancestors() const;
    [[nodiscard]] std::vector<Particle> descendants() const;

    [[nodiscard]] bool hasAncestorPdgId(int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstAncestorWithPdgId(int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstCommonAncestor(Particle other) const;

    [[nodiscard]] bool operator==(Particle const& other) const { return graph_ == other.graph_ && id_ == other.id_; }
    [[nodiscard]] bool operator!=(Particle const& other) const { return !(*this == other); }

  private:
    Graph const* graph_ = nullptr;
    uint32_t id_ = 0;
  };

}  // namespace truth

#endif
