#ifndef PrimaryVertexSorter_H
#define PrimaryVertexSorter_H

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <vector>

/** \class PrimaryVertexSorter
 * class to sort VertexCollection
 * in decreasing order of the sum of the squared track pT's
 */
struct PrimaryVertexSorter {

  std::vector<reco::Vertex> sortedList(const reco::VertexCollection & primaryVertex) const;

};

#endif
