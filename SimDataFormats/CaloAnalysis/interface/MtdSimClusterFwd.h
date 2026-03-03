#ifndef CaloAnalysis_MtdSimClusterFwd_h
#define CaloAnalysis_MtdSimClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace io_v1 {
  class MtdSimCluster;
  std::ostream &operator<<(std::ostream &s, MtdSimCluster const &tp);
}  // namespace io_v1
using MtdSimCluster = io_v1::MtdSimCluster;

typedef std::vector<MtdSimCluster> MtdSimClusterCollection;
typedef edm::Ref<MtdSimClusterCollection> MtdSimClusterRef;
typedef edm::RefVector<MtdSimClusterCollection> MtdSimClusterRefVector;
typedef edm::RefProd<MtdSimClusterCollection> MtdSimClusterRefProd;
typedef edm::RefVector<MtdSimClusterCollection> MtdSimClusterContainer;

#endif
