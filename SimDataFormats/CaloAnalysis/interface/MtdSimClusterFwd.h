#ifndef CaloAnalysis_MtdSimClusterFwd_h
#define CaloAnalysis_MtdSimClusterFwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

class MtdSimCluster;
typedef std::vector<MtdSimCluster> MtdSimClusterCollection;
typedef edm::Ref<MtdSimClusterCollection> MtdSimClusterRef;
typedef edm::RefVector<MtdSimClusterCollection> MtdSimClusterRefVector;
typedef edm::RefProd<MtdSimClusterCollection> MtdSimClusterRefProd;
typedef edm::RefVector<MtdSimClusterCollection> MtdSimClusterContainer;

std::ostream &operator<<(std::ostream &s, MtdSimCluster const &tp);

#endif
