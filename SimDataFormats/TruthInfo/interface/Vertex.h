// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimDataFormats_TruthInfo_interface_Vertex_h
#define SimDataFormats_TruthInfo_interface_Vertex_h

#include <cstdint>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/VertexData.h"

namespace truth {

  class Graph;
  class Particle;

  // A lightweight, copyable view of one logical vertex. It stores only a graph
  // pointer and an id; all accessors read through to the owning Graph (the heavy
  // method bodies live in Graph.cc).
  class Vertex {
  public:
    Vertex() = default;
    Vertex(Graph const* graph, uint32_t id) : graph_(graph), id_(id) {}

    [[nodiscard]] bool valid() const { return graph_ != nullptr; }
    [[nodiscard]] uint32_t id() const { return id_; }

    [[nodiscard]] const VertexData& data() const;

    [[nodiscard]] bool hasGen() const;
    [[nodiscard]] bool hasSim() const;
    [[nodiscard]] uint64_t eventId() const;
    [[nodiscard]] int32_t genEvent() const;
    [[nodiscard]] const math::XYZTLorentzVectorD& position() const;

    [[nodiscard]] bool isSource() const;
    [[nodiscard]] bool isSink() const;

    [[nodiscard]] std::vector<Particle> incomingParticles() const;
    [[nodiscard]] std::vector<Particle> outgoingParticles() const;

    [[nodiscard]] bool operator==(Vertex const& other) const { return graph_ == other.graph_ && id_ == other.id_; }
    [[nodiscard]] bool operator!=(Vertex const& other) const { return !(*this == other); }

  private:
    Graph const* graph_ = nullptr;
    uint32_t id_ = 0;
  };

}  // namespace truth

#endif
