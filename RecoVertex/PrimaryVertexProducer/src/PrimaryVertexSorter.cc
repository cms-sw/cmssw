#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"

using namespace reco;

VertexCollection
PrimaryVertexSorter::sortedList(const VertexCollection & unsortedPVColl) const
{
  VertexCollection pvs = unsortedPVColl;
  sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
  return pvs;
}
