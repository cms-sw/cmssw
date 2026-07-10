#ifndef CaloAnalysis_SimClusterFwd_h
#define CaloAnalysis_SimClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace io_v1 {
  class SimCluster;
  std::ostream &operator<<(std::ostream &s, SimCluster const &tp);
}  // namespace io_v1
using SimCluster = io_v1::SimCluster;

typedef std::vector<SimCluster> SimClusterCollection;
typedef edm::Ref<SimClusterCollection> SimClusterRef;
typedef edm::RefVector<SimClusterCollection> SimClusterRefVector;
typedef edm::RefProd<SimClusterCollection> SimClusterRefProd;
typedef edm::RefVector<SimClusterCollection> SimClusterContainer;

#endif
