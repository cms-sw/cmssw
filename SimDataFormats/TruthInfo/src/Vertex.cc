// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "SimDataFormats/TruthInfo/interface/Vertex.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

const truth::VertexData& truth::Vertex::data() const {
  // See Particle::data(): return a shared empty record for an invalid view rather
  // than dereferencing a null graph_.
  if (graph_ == nullptr) {
    static const truth::VertexData kEmpty{};
    return kEmpty;
  }
  return graph_->vertices_.at(id_);
}

bool truth::Vertex::hasGen() const { return data().hasGen(); }

bool truth::Vertex::hasSim() const { return data().hasSim(); }

uint64_t truth::Vertex::eventId() const { return data().eventId; }

int32_t truth::Vertex::genEvent() const { return data().genEvent; }

bool truth::Vertex::isSignal() const { return valid() && data().isSignal(); }

bool truth::Vertex::isFromPileup() const { return valid() && !data().isSignal(); }

const math::XYZTLorentzVectorD& truth::Vertex::position() const { return data().position; }

bool truth::Vertex::isSource() const { return valid() && graph_->incomingParticles(id_).empty(); }

bool truth::Vertex::isSink() const { return valid() && graph_->outgoingParticles(id_).empty(); }

std::vector<truth::Particle> truth::Vertex::incomingParticles() const {
  return valid() ? graph_->incomingParticlesOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Vertex::outgoingParticles() const {
  return valid() ? graph_->outgoingParticlesOf(id_) : std::vector<truth::Particle>{};
}
