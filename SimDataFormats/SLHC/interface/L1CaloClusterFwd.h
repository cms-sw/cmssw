#ifndef L1CaloCluster_Fwd
#define L1caloCLuster_Fwd
#include "DataFormats/Common/interface/Ref.h"

namespace l1slhc {
class L1CaloCluster;
}

namespace l1slhc {
  typedef std::vector<L1CaloCluster> L1CaloClusterCollection;
  typedef edm::Ref<L1CaloClusterCollection> L1CaloClusterRef;
}

#endif
