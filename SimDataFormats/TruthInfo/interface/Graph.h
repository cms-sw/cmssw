// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimDataFormats_TruthInfo_interface_Graph_h
#define SimDataFormats_TruthInfo_interface_Graph_h

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/Checkpoint.h"
#include "SimDataFormats/TruthInfo/interface/ParticleData.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

namespace truth {

  // The user-facing logical truth graph: a bipartite Particle <-> Vertex graph
  // stored CSR-style. Built once (the producer / post-processor fill the storage
  // via the non-const accessors) and read many times through the navigation API
  // and the lightweight Particle / Vertex views.
  class Graph {
  public:
    using size_type = uint32_t;

    // --- Storage accessors --------------------------------------------------
    // Built once (the producer / post-processor fill via the non-const accessors)
    // and read many times (consumers use the const accessors and the navigation
    // API below). The bipartite adjacency is stored CSR-style: each *Offsets vector
    // has size n+1 and indexes into the matching flat target vector.
    [[nodiscard]] std::vector<ParticleData> const& particles() const { return particles_; }
    [[nodiscard]] std::vector<ParticleData>& particles() { return particles_; }

    [[nodiscard]] std::vector<VertexData> const& vertices() const { return vertices_; }
    [[nodiscard]] std::vector<VertexData>& vertices() { return vertices_; }

    [[nodiscard]] std::vector<uint32_t> const& particleToDecayVertexOffsets() const {
      return particleToDecayVertexOffsets_;
    }
    [[nodiscard]] std::vector<uint32_t>& particleToDecayVertexOffsets() { return particleToDecayVertexOffsets_; }

    [[nodiscard]] std::vector<uint32_t> const& particleToDecayVertices() const { return particleToDecayVertices_; }
    [[nodiscard]] std::vector<uint32_t>& particleToDecayVertices() { return particleToDecayVertices_; }

    [[nodiscard]] std::vector<uint32_t> const& particleToProductionVertexOffsets() const {
      return particleToProductionVertexOffsets_;
    }
    [[nodiscard]] std::vector<uint32_t>& particleToProductionVertexOffsets() {
      return particleToProductionVertexOffsets_;
    }

    [[nodiscard]] std::vector<uint32_t> const& particleToProductionVertices() const {
      return particleToProductionVertices_;
    }
    [[nodiscard]] std::vector<uint32_t>& particleToProductionVertices() { return particleToProductionVertices_; }

    [[nodiscard]] std::vector<uint32_t> const& vertexToOutgoingParticleOffsets() const {
      return vertexToOutgoingParticleOffsets_;
    }
    [[nodiscard]] std::vector<uint32_t>& vertexToOutgoingParticleOffsets() { return vertexToOutgoingParticleOffsets_; }

    [[nodiscard]] std::vector<uint32_t> const& vertexToOutgoingParticles() const { return vertexToOutgoingParticles_; }
    [[nodiscard]] std::vector<uint32_t>& vertexToOutgoingParticles() { return vertexToOutgoingParticles_; }

    [[nodiscard]] std::vector<uint32_t> const& vertexToIncomingParticleOffsets() const {
      return vertexToIncomingParticleOffsets_;
    }
    [[nodiscard]] std::vector<uint32_t>& vertexToIncomingParticleOffsets() { return vertexToIncomingParticleOffsets_; }

    [[nodiscard]] std::vector<uint32_t> const& vertexToIncomingParticles() const { return vertexToIncomingParticles_; }
    [[nodiscard]] std::vector<uint32_t>& vertexToIncomingParticles() { return vertexToIncomingParticles_; }

    [[nodiscard]] size_type nParticles() const { return static_cast<size_type>(particles_.size()); }
    [[nodiscard]] size_type nVertices() const { return static_cast<size_type>(vertices_.size()); }

    [[nodiscard]] bool empty() const { return particles_.empty() && vertices_.empty(); }

    [[nodiscard]] Particle particle(size_type id) const;
    [[nodiscard]] Vertex vertex(size_type id) const;

    [[nodiscard]] std::vector<Particle> particleViews() const;
    [[nodiscard]] std::vector<Vertex> vertexViews() const;

    [[nodiscard]] std::vector<Particle> roots() const;
    [[nodiscard]] std::vector<Particle> leaves() const;

    // Lowest (closest) common ancestor of a set of particles: the single truth
    // particle from which all of them descend, minimizing the total number of
    // generations. This answers "which particle did this jet come from" given
    // the jet's truth constituents (e.g. the b quark of a b-jet); walk further
    // up with Particle::firstAncestorWithPdgId to reach a specific origin
    // species (e.g. the top). Returns nullopt if the inputs share no ancestor.
    [[nodiscard]] std::optional<Particle> lowestCommonAncestor(std::vector<Particle> const& particles) const;

    [[nodiscard]] std::vector<Vertex> sourceVertices() const;
    [[nodiscard]] std::vector<Vertex> sinkVertices() const;

    [[nodiscard]] std::span<const uint32_t> decayVertices(size_type particleId) const {
      const auto begin = particleToDecayVertexOffsets_.at(particleId);
      const auto end = particleToDecayVertexOffsets_.at(particleId + 1);
      return std::span<const uint32_t>(particleToDecayVertices_.data() + begin, end - begin);
    }

    [[nodiscard]] std::span<const uint32_t> productionVertices(size_type particleId) const {
      const auto begin = particleToProductionVertexOffsets_.at(particleId);
      const auto end = particleToProductionVertexOffsets_.at(particleId + 1);
      return std::span<const uint32_t>(particleToProductionVertices_.data() + begin, end - begin);
    }

    [[nodiscard]] std::span<const uint32_t> outgoingParticles(size_type vertexId) const {
      const auto begin = vertexToOutgoingParticleOffsets_.at(vertexId);
      const auto end = vertexToOutgoingParticleOffsets_.at(vertexId + 1);
      return std::span<const uint32_t>(vertexToOutgoingParticles_.data() + begin, end - begin);
    }

    [[nodiscard]] std::span<const uint32_t> incomingParticles(size_type vertexId) const {
      const auto begin = vertexToIncomingParticleOffsets_.at(vertexId);
      const auto end = vertexToIncomingParticleOffsets_.at(vertexId + 1);
      return std::span<const uint32_t>(vertexToIncomingParticles_.data() + begin, end - begin);
    }

    [[nodiscard]] bool isConsistent() const;

  private:
    friend class Particle;
    friend class Vertex;

    [[nodiscard]] std::vector<Vertex> productionVerticesOf(size_type particleId) const;
    [[nodiscard]] std::vector<Vertex> decayVerticesOf(size_type particleId) const;

    [[nodiscard]] std::vector<Particle> parentsOf(size_type particleId) const;
    [[nodiscard]] std::vector<Particle> childrenOf(size_type particleId) const;

    // Allocation-free cores for the traversals: append the immediate parent/child
    // particle ids (excluding self) to `out`, without deduplication. Callers that
    // need unique results dedup against their own (tiny) buffer or a visited set.
    void appendParents(size_type particleId, std::vector<uint32_t>& out) const;
    void appendChildren(size_type particleId, std::vector<uint32_t>& out) const;

    [[nodiscard]] std::vector<Particle> ancestorsOf(size_type particleId) const;
    [[nodiscard]] std::vector<Particle> descendantsOf(size_type particleId) const;

    [[nodiscard]] std::optional<Particle> firstAncestorWithPdgIdOf(size_type particleId, int pdgId) const;
    [[nodiscard]] std::optional<Particle> firstCommonAncestorOf(size_type a, size_type b) const;

    [[nodiscard]] std::vector<Particle> incomingParticlesOf(size_type vertexId) const;
    [[nodiscard]] std::vector<Particle> outgoingParticlesOf(size_type vertexId) const;

    std::vector<ParticleData> particles_;
    std::vector<VertexData> vertices_;

    // Particle -> decay vertices
    std::vector<uint32_t> particleToDecayVertexOffsets_;
    std::vector<uint32_t> particleToDecayVertices_;

    // Particle -> production vertices
    std::vector<uint32_t> particleToProductionVertexOffsets_;
    std::vector<uint32_t> particleToProductionVertices_;

    // Vertex -> outgoing particles
    std::vector<uint32_t> vertexToOutgoingParticleOffsets_;
    std::vector<uint32_t> vertexToOutgoingParticles_;

    // Vertex -> incoming particles
    std::vector<uint32_t> vertexToIncomingParticleOffsets_;
    std::vector<uint32_t> vertexToIncomingParticles_;
  };

}  // namespace truth

#endif
