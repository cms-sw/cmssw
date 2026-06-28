// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Author: Felice Pantaleo - CERN
// Date: 03/2026
// A compact, read-only graph representation of the truth information in an event.
// The graph is built in the TruthGraphProducer module, which also fills the node metadata and associations.
// The graph is intended to be a common data format for various use cases (e.g. validation, analysis, visualization).

#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

bool TruthGraph::isConsistent() const {
  if (offsets_.size() != nodes_.size() + 1)
    return false;
  if (!offsets_.empty() && offsets_.front() != 0)
    return false;
  if (!offsets_.empty() && offsets_.back() != edges_.size())
    return false;
  if (!edgeKind_.empty() && edgeKind_.size() != edges_.size())
    return false;

  for (size_t i = 1; i < offsets_.size(); ++i) {
    if (offsets_[i] < offsets_[i - 1])
      return false;
  }
  return true;
}
