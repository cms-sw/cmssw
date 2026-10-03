// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "HepPDT/ParticleID.hh"

const truth::ParticleData& truth::Particle::data() const {
  // Graph::particle() returns an invalid (null-graph) view for an out-of-range id,
  // and a default-constructed Particle is likewise invalid. The scalar getters all
  // route through data(), so return a shared empty record instead of dereferencing
  // a null graph_ (the traversal methods already guard with valid()).
  if (graph_ == nullptr) {
    static const truth::ParticleData kEmpty{};
    return kEmpty;
  }
  return graph_->particles_.at(id_);
}

bool truth::Particle::hasGen() const { return data().hasGen(); }

bool truth::Particle::hasSim() const { return data().hasSim(); }

int32_t truth::Particle::pdgId() const { return data().pdgId; }

int16_t truth::Particle::status() const { return data().status; }

uint64_t truth::Particle::eventId() const { return data().eventId; }

int32_t truth::Particle::genEvent() const { return data().genEvent; }

const math::XYZTLorentzVectorD& truth::Particle::momentum() const { return data().momentum; }

double truth::Particle::charge() const { return HepPDT::ParticleID(data().pdgId).charge(); }

bool truth::Particle::isSignal() const { return valid() && data().isSignal(); }

bool truth::Particle::isFromPileup() const { return valid() && !data().isSignal(); }

std::span<const truth::Checkpoint> truth::Particle::checkpoints() const {
  return std::span<const truth::Checkpoint>(data().checkpoints.data(), data().checkpoints.size());
}

bool truth::Particle::hasCheckpoints() const { return !data().checkpoints.empty(); }

std::optional<truth::Checkpoint> truth::Particle::checkpoint(uint32_t checkpointId) const {
  for (auto const& cp : data().checkpoints) {
    if (cp.checkpointId == checkpointId)
      return cp;
  }
  return std::nullopt;
}

uint16_t truth::Particle::statusFlags() const { return data().statusFlags; }

bool truth::Particle::backscattered() const { return data().backscattered; }

bool truth::Particle::isRoot() const { return valid() && graph_->productionVertices(id_).empty(); }

bool truth::Particle::isLeaf() const { return valid() && graph_->decayVertices(id_).empty(); }

std::vector<truth::Vertex> truth::Particle::productionVertices() const {
  return valid() ? graph_->productionVerticesOf(id_) : std::vector<truth::Vertex>{};
}

std::vector<truth::Vertex> truth::Particle::decayVertices() const {
  return valid() ? graph_->decayVerticesOf(id_) : std::vector<truth::Vertex>{};
}

std::vector<truth::Particle> truth::Particle::parents() const {
  return valid() ? graph_->parentsOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Particle::children() const {
  return valid() ? graph_->childrenOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Particle::ancestors() const {
  return valid() ? graph_->ancestorsOf(id_) : std::vector<truth::Particle>{};
}

uint32_t truth::Particle::ancestorCount() const { return valid() ? graph_->ancestorCount(id_) : 0; }

std::vector<truth::Particle> truth::Particle::descendants() const {
  return valid() ? graph_->descendantsOf(id_) : std::vector<truth::Particle>{};
}

bool truth::Particle::hasAncestorPdgId(int pdgId) const { return firstAncestorWithPdgId(pdgId).has_value(); }

std::optional<truth::Particle> truth::Particle::firstAncestorWithPdgId(int pdgId) const {
  return valid() ? graph_->firstAncestorWithPdgIdOf(id_, pdgId) : std::nullopt;
}

truth::Particle truth::Particle::lastCopy() const {
  return valid() ? Particle(graph_, graph_->lastCopyOf(id_)) : *this;
}

std::optional<truth::Particle> truth::Particle::firstChildWithPdgId(int pdgId) const {
  return valid() ? graph_->firstChildWithPdgIdOf(id_, pdgId) : std::nullopt;
}

std::vector<truth::Particle> truth::Particle::productionSiblings() const {
  return valid() ? graph_->productionSiblingsOf(id_) : std::vector<truth::Particle>{};
}

std::optional<truth::Particle> truth::Particle::firstCommonAncestor(Particle other) const {
  if (!valid() || !other.valid() || graph_ != other.graph_)
    return std::nullopt;
  return graph_->firstCommonAncestorOf(id_, other.id_);
}
