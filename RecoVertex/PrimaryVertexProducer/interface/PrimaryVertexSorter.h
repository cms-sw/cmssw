#ifndef PrimaryVertexSorter_H
#define PrimaryVertexSorter_H

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <vector>

/** \class PrimaryVertexSorter
 * operator for sorting TransientVertex objects
 * in decreasing order of the sum of the squared track pT's
 */
struct PrimaryVertexSorter {

  std::vector<reco::Vertex> sortedList(reco::VertexCollection primaryVertex) const;

//   bool operator() ( const TransientVertex & v1, 
// 		    const TransientVertex & v2) const;
// 
// 
// private:
// 
//   double sumPtSquared(const std::vector<reco::TransientTrack> & tks) const;
};

#endif
