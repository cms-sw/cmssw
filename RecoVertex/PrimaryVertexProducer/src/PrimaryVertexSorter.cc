#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"

using namespace reco;
using namespace std;

vector<reco::Vertex>
PrimaryVertexSorter::sortedList(VertexCollection unsortedPVColl) const
{
  vector<Vertex> pvs;
  pvs.reserve(unsortedPVColl.size());
  for (VertexCollection::size_type i = 0; i < unsortedPVColl.size(); ++i) {
    pvs.push_back(unsortedPVColl[i]);
  }

    // sort vertices by pt**2  vertex (aka signal vertex tagging)
  sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());

  return pvs;
}
