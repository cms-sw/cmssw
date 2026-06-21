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
  if (offsets.size() != nodes.size() + 1)
    return false;
  if (!offsets.empty() && offsets.front() != 0)
    return false;
  if (!offsets.empty() && offsets.back() != edges.size())
    return false;
  if (!edgeKind.empty() && edgeKind.size() != edges.size())
    return false;

  for (size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i] < offsets[i - 1])
      return false;
  }
  return true;
}
